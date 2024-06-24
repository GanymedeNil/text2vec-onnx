import numpy as np
from typing import List, Union
from tokenizers import Tokenizer
from pathlib import Path
from enum import Enum

try:
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_available_providers, \
        get_device
except:
    print("please pip3 install onnxruntime")

import warnings


class EncoderType(Enum):
    LAST_AVG = 0
    CLS = 1
    MEAN = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class SentenceModel:
    def __init__(self, model_dir_path: str, encoder_type: Union[str, EncoderType] = "MEAN", device_id=-1,
                 intra_op_num_threads=4, max_seq_length=256, ):
        """
                Initializes the base sentence model.

                :param model_dir_path: The directory path of the model.
                :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
                       LAST_AVG, CLS, MEAN(mean of last_hidden_state)
                :param max_seq_length: The maximum sequence length.
                :param device_id: The device id to use, -1 for CPU, 0 for GPU.
                :param intra_op_num_threads: The onnxruntime intra_op_num_threads.

                """
        self.max_seq_length = max_seq_length
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f"encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type

        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = 'CUDAExecutionProvider'
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = 'CPUExecutionProvider'
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if device_id != "-1" and get_device() == 'GPU' \
                and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))
        print(EP_list, get_device())
        tokenizer_path = f"{model_dir_path}/tokenizer.json"
        self._verify_model(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        model_path = f"{model_dir_path}/model.onnx"
        self._verify_model(model_path)
        self.model = InferenceSession(model_path, sess_options=sess_opt,
                                      providers=EP_list)
        if device_id != "-1" and cuda_ep not in self.model.get_providers():
            warnings.warn(
                f'{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n'
                'Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, '
                'you can check their relations from the offical web site: '
                'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html',
                RuntimeWarning)

    def _token_encode(self, sentences: Union[str, List[str]], max_seq_length):
        if isinstance(sentences, str):
            sentences = [sentences]
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        batches = self.tokenizer.encode_batch(sentences)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for item in batches:
            input_ids.append(np.array(item.ids))
            attention_mask.append(np.array(item.attention_mask))
            token_type_ids.append(np.array(item.type_ids))
        return dict(input_ids=np.array(input_ids), token_type_ids=np.array(token_type_ids),
                    attention_mask=np.array(attention_mask))

    def get_sentence_embeddings(self, inputs: dict):
        token_embeddings = self.model.run(output_names=["last_hidden_state"], input_feed=inputs)

        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = token_embeddings[0]
            final_encoding = np.mean(sequence_output, axis=1)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = token_embeddings[0]
            return sequence_output[:, 0]

        if self.encoder_type == EncoderType.MEAN:
            attention_mask = inputs["attention_mask"]
            attention_mask = np.array(attention_mask)
            token_embeddings = np.array(token_embeddings[0])

            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).repeat(token_embeddings.shape[-1], axis=-1)
            input_mask_expanded = input_mask_expanded.astype(float)

            masked_embeddings = token_embeddings * input_mask_expanded

            sum_masked_embeddings = np.sum(masked_embeddings, axis=1)
            sum_input_mask = np.sum(input_mask_expanded, axis=1)
            sum_input_mask = np.clip(sum_input_mask, a_min=1e-9, a_max=None)

            final_encoding = sum_masked_embeddings / sum_input_mask
            return final_encoding

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, max_seq_length: int = None):
        """
               Returns the embeddings for a batch of sentences.

               :param sentences: str/list, Input sentences
               :param batch_size: int, Batch size
               :param max_seq_length: Override value for max_seq_length
               """
        all_embeddings = []

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            features = self._token_encode(sentences_batch, max_seq_length)
            embeddings = self.get_sentence_embeddings(features)
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray(all_embeddings)
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exists.')
        if not model_path.is_file():
            raise FileExistsError(f'{model_path} is not a file.')
