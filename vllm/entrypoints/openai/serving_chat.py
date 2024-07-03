import codecs
import time
import re
import random
from dataclasses import dataclass, field
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Dict, Iterable,
                    List, Optional)
from typing import Sequence as GenericSequence
from typing import TypedDict, Union, cast, final

from fastapi import Request
from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartTextParam)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionContentPartParam, ChatCompletionLogProb,
    ChatCompletionLogProbs, ChatCompletionLogProbsContent,
    ChatCompletionMessageParam, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    FunctionCall, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.utils import (async_get_and_parse_image,
                                   get_full_image_text_prompt)
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.utils import random_uuid

logger = init_logger(__name__)


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    mm_futures: List[Awaitable[MultiModalDataDict]] = field(
        default_factory=list)


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 response_role: str,
                 lora_modules: Optional[List[LoRAModulePath]] = None,
                 chat_template: Optional[str] = None):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)

        self.response_role = response_role
        self._load_chat_template(chat_template)

    def _load_chat_template(self, chat_template: Optional[str]):
        tokenizer = self.tokenizer

        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError as e:
                JINJA_CHARS = "{}\n"
                if not any(c in chat_template for c in JINJA_CHARS):
                    msg = (f"The supplied chat template ({chat_template}) "
                           f"looks like a file path, but it failed to be "
                           f"opened. Reason: {e}")
                    raise ValueError(msg) from e

                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info("Using supplied chat template:\n%s",
                        tokenizer.chat_template)
        elif tokenizer.chat_template is not None:
            logger.info("Using default chat template:\n%s",
                        tokenizer.chat_template)
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")

    def _parse_chat_message_content_parts(
        self,
        role: str,
        parts: Iterable[ChatCompletionContentPartParam],
    ) -> ChatMessageParseResult:
        texts: List[str] = []
        mm_futures: List[Awaitable[MultiModalDataDict]] = []

        vlm_config: Optional[VisionLanguageConfig] = getattr(
            self.engine.engine, "vision_language_config", None)
        model_config = getattr(self.engine.engine, "model_config", None)

        for part in parts:
            part_type = part["type"]
            if part_type == "text":
                text = cast(ChatCompletionContentPartTextParam, part)["text"]
                texts.append(text)
            elif part_type == "image_url":
                if vlm_config is None:
                    raise ValueError(
                        "'image_url' input is not supported as the loaded "
                        "model is not multimodal.")
                assert self.tokenizer is not None
                image_url = cast(ChatCompletionContentPartImageParam,
                                 part)["image_url"]

                if image_url.get("detail", "auto") != "auto":
                    logger.warning(
                        "'image_url.detail' is currently not supported and "
                        "will be ignored.")

                mm_future = async_get_and_parse_image(image_url["url"])
                mm_futures.append(mm_future)

            else:
                raise NotImplementedError(f"Unknown part type: {part_type}")

        text_prompt = "\n".join(texts)

        if vlm_config is not None and len(mm_futures):

            assert len(
                mm_futures
            ) == 1, "Multiple 'image_url' input is currently not supported."
            (image_token_prompt,
             image_token_str) = vlm_config.get_image_token_text(self.tokenizer)

            # NOTE: If image token string (e.g, <image>) is already present
            # in the text prompt, we assume it follows the same format required
            # by the engine.
            if image_token_str in text_prompt:
                logger.warning(
                    "Detected image token string in the text prompt. "
                    "Skipping prompt formatting.")
                messages = [
                    ConversationMessage(role=role, content=text_prompt)
                ]

            else:
                full_prompt = get_full_image_text_prompt(
                    image_prompt=image_token_prompt,
                    text_prompt=text_prompt,
                    config=model_config)
                messages = [
                    ConversationMessage(role=role, content=full_prompt)
                ]
        else:
            messages = [ConversationMessage(role=role, content=text_prompt)]

        return ChatMessageParseResult(messages=messages, mm_futures=mm_futures)

    def _parse_chat_message_content(
        self,
        message: ChatCompletionMessageParam,
    ) -> ChatMessageParseResult:
        role = message["role"]
        content = message.get("content")

        if content is None:
            return ChatMessageParseResult(messages=[], mm_futures=[])
        if isinstance(content, str):
            messages = [ConversationMessage(role=role, content=content)]
            return ChatMessageParseResult(messages=messages, mm_futures=[])

        return self._parse_chat_message_content_parts(role, content)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        try:
            conversation: List[ConversationMessage] = []
            mm_futures: List[Awaitable[MultiModalDataDict]] = []

            for msg in request.messages:
                chat_parsed_result = self._parse_chat_message_content(msg)

                conversation.extend(chat_parsed_result.messages)
                mm_futures.extend(chat_parsed_result.mm_futures)

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
        except Exception as e:
            logger.error("Error in applying chat template from request: %s", e)
            return self.create_error_response(str(e))

        mm_data: Optional[MultiModalDataDict] = None
        try:
            if len(mm_futures):
                # since we support only single mm data currently
                assert len(mm_futures) == 1
                mm_data = await mm_futures[0]
        except Exception as e:
            logger.error("Error in loading multi-modal data: %s", e)
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        try:
            # Tokenize/detokenize depending on prompt format (string/token list)
            prompt_ids, prompt_text = self._validate_prompt_and_tokenize(
                request,
                prompt=prompt,
                add_special_tokens=request.add_special_tokens)
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            decoding_config = await self.engine.get_decoding_config()
            guided_decoding_backend = request.guided_decoding_backend \
                or decoding_config.guided_decoding_backend
            guided_decode_logits_processor = (
                await get_guided_decoding_logits_processor(
                    guided_decoding_backend, request, await
                    self.engine.get_tokenizer()))
            if guided_decode_logits_processor:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logits_processor)
        except ValueError as e:
            return self.create_error_response(str(e))

        inputs: PromptInputs = {
            "prompt": prompt_text,
            "prompt_token_ids": prompt_ids,
        }
        if mm_data is not None:
            inputs["multi_modal_data"] = mm_data

        is_tracing_enabled = await self.engine.is_tracing_enabled()
        trace_headers = None
        if is_tracing_enabled and raw_request:
            trace_headers = extract_trace_headers(raw_request.headers)
        if not is_tracing_enabled and raw_request and contains_trace_headers(
                raw_request.headers):
            log_tracing_disabled_warning()

        result_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id,
            lora_request,
            trace_headers=trace_headers,
        )
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation)
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, result_generator, request_id,
                    conversation)
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str,
            conversation: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        model_name = self.served_model_names[0]
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        assert request.n is not None
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        try:
            async for res in result_generator:
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    for i in range(request.n):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name)
                                if (request.stream_options and
                                        request.stream_options.include_usage):
                                    chunk.usage = None
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    out_logprobs = output.logprobs[
                        previous_num_tokens[i]:] if output.logprobs else None

                    if request.logprobs and request.top_logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.top_logprobs,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    if request.tool_choice and type(
                            request.tool_choice
                    ) is ChatCompletionNamedToolChoiceParam:
                        delta_message = DeltaMessage(tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text))
                        ])
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n

                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True

            if (request.stream_options
                    and request.stream_options.include_usage):
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=previous_num_tokens[i],
                    total_tokens=prompt_tokens + previous_num_tokens[i],
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self, request: ChatCompletionRequest, raw_request: Optional[Request],
        result_generator: AsyncIterator[RequestOutput], request_id: str,
        conversation: List[ConversationMessage]
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.served_model_names[0]
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        print("n:{}\npresence_penalty:{}\nfrequency_penalty:{}\nrepetition_penalty:{}\ntemperature:{}\ntop_p:{}\nmin_p:{}\nstop:{}\nstop_token_ids:{}\nmax_tokens:{}\nbest_of:{}\ntop_k:{}\nignore_eos:{}\nuse_beam_search:{}\nskip_special_tokens:{}\nspaces_between_special_tokens:{}\n".format(request.n, request.presence_penalty, request.frequency_penalty, request.repetition_penalty, request.temperature, request.top_p, request.min_p, request.stop, request.stop_token_ids, request.max_tokens, request.best_of, request.top_k, request.ignore_eos, request.use_beam_search, request.skip_special_tokens, request.spaces_between_special_tokens))
        
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
            
            # 超出max_tokens主动截止在终止符
            for output in final_res.outputs:
                if len(output.token_ids) > request.max_tokens_temp:    
                    endings = ('\n', '!', '.', '?', '~', ')', '*')

                    if output.text.endswith(endings):
                        await self.engine.abort(request_id)
        
        assert final_res is not None
        
        output.text = output.text.strip()
        print(f"原文:\n{output.text}")
        
        choices = []
        outputs = []
        
        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            print(f"\nmax tokens: {request.max_tokens_temp}")
            print(f"prompt tokens: {len(final_res.prompt_token_ids)}")
            print(f"output tokens: {len(output.token_ids)}")

            rate = len(output.text.split(" ")) / len(output.token_ids)
            print(f"words:tokens={rate}")

            #************ 去除末尾不完整内容  *************
            endIndex = -1
            
            for (i, char) in enumerate(output.text[::-1]):
                if (char == ".") or (char == "!") or (char == "?") or (char == "~") or (char == ")"):# or (char == "*"):
                    endIndex = len(output.text) - i
                    break
            
            
            if endIndex == -1:
                # 没有终止符并且句子太短
                if len(output.text) <= 10:
                    output.text = ""
            else:
                # 有终止符，在之后的内容要截断
                message = output.text[:endIndex]
                outContent = output.text[endIndex:]

                """
                eofColon = ''
                eofColonIndex = message.rfind('"')

                if len(message) > (eofColonIndex+1) and (eofColonIndex != -1):
                    if re.search("[a-zA-Z]", message[eofColonIndex+1]): 
                        eofColon = '"'

                eofAsterisk = ''
                eofAsteriskIndex = message.rfind('*')

                if len(message) > (eofAsteriskIndex+1) and (eofAsteriskIndex != -1):
                    if re.search("[a-zA-Z]", message[eofAsteriskIndex+1]): 
                        eofAsterisk = '*'
                """

                output.text = message # + eofColon + eofAsterisk #显示被去除的内容 + "  >>>>>  " + outContent
            #************ 去除末尾不完整内容 end   ***************

            #************ fix markdown   ***************
            def fix_markdown(text: str, for_display: bool) -> str:
                def replace_spaces(match):
                    # 去除格式字符周围的空格
                    return match.group(1) + match.group(2).strip() + match.group(1)

                # 匹配格式化标记及其包围的内容
                format_pattern = re.compile(r'([*_]{1,2})(.*?)\1')
                # 逆向替换，避免索引问题
                matches = list(re.finditer(format_pattern, text))
                for match in reversed(matches):
                    start, end = match.span()
                    text = text[:start] + replace_spaces(match) + text[end:]

                if for_display:
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        # 修正未成对的格式字符和引号
                        for char in ['*', '"']:
                            if line.count(char) % 2 != 0:
                                lines[i] = line.strip() + char
                    text = '\n'.join(lines)

                return text

            markdown = fix_markdown(output.text, True)
            if markdown != output.text:
                print("尝试修复markdown")
            output.text = markdown
            #************ fix markdown end   ***************

            print(f"\n截断和补全:\n{output.text}")

            #************ 统计结尾词 **************

            """
            cleaned_string = re.sub(r"[^,、\s]", "", output.text)
            if cleaned_string != cleaned_string.replace(","*15, "") or cleaned_string != cleaned_string.replace("、"*15, "") or cleaned_string != cleaned_string.replace(";"*15, "") or cleaned_string != cleaned_string.replace(" "*50, ""):
                print("词穷了")
                output.text = ""

            eos_words = ["As decades roll by", "Years passed", "legend folklore", "End of Story", "story by writing", "Years later", "decades"]
            for eos_word in eos_words:
                if eos_word in output.text:
                    output.text = ""
                    print("匹配到eos词")
                    print(eos_word)
                    continue
            """

            print(f"\n最终:\n{output.text}")


            def process_tts_text(text):
                # 假设 skip_codeblocks 设置为 True
                text = re.sub(r'^\s{4}.*$', '', text, flags=re.MULTILINE).strip()
                text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()

                # 假设 skip_tags 设置为 True
                text = re.sub(r'<.*?>.*?</.*?>', '', text, flags=re.DOTALL).strip()

                # 假设 pass_asterisks 设置为 False
                text = re.sub(r'\*[^*]*?(\*|$)', '', text).strip()  # remove asterisks content

                # 假设 narrate_quoted_only 设置为 True
                text = re.sub(r'[“”«»]', '"', text)  # Normalize special quotes to standard quotes
                matches = re.findall(r'"[^"]*"', text)  # Matches text inside double quotes, non-greedily
                text = ' ... '.join(matches) if matches else text

                # Replace fancy ellipsis with "..."
                text = text.replace('…', '...')
                # Remove quotes
                text = text.replace('"', '').replace('“', '').replace('”', '').replace('‘', '').replace('’', '')
                # Replace multiple "." with single "."
                text = re.sub(r'\.+', '.', text)

                # Collapse newlines and spaces into single space
                text = re.sub(r'\s+', ' ', text).strip()

                print(f'\nTTS: {text}')

                return text
            
            #process_tts_text(output.text)
            
            if output.text == "":
                continue
            if output.text in outputs:
                continue

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)
            outputs.append(output.text)

        if request.echo:
            last_msg_content = ""
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == role:
                last_msg_content = conversation[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    def _get_top_logprobs(
            self, logprobs: Dict[int, Logprob],
            top_logprobs: Optional[int]) -> List[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=self._get_decoded_token(p[1], p[0]),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(
                    self._get_decoded_token(p[1],
                                            p[0]).encode("utf-8",
                                                         errors="replace")))
            for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        num_output_top_logprobs: Optional[int] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""

        logprobs_content = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self.tokenizer.decode(token_id),
                        bytes=list(
                            self.tokenizer.decode(token_id).encode(
                                "utf-8", errors="replace"))))
            else:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=step_top_logprobs[token_id].decoded_token,
                        logprob=max(step_top_logprobs[token_id].logprob,
                                    -9999.0),
                        bytes=list(
                            step_top_logprobs[token_id].decoded_token.encode(
                                "utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs, num_output_top_logprobs)))

        return ChatCompletionLogProbs(content=logprobs_content)
