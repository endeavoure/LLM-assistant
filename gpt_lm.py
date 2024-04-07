import transformers
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, Sequence, List, Iterator, Tuple

class GPTWrapper:

    def __init__(self):

        self.model = transformers.GPT2LMHeadModel.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32)
        # self.model.to("mps") p.s. ошибка прошлого))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"})

    def generate(self, input_text, **generation_kwargs):
        # self.model.to("mps") p.s. ошибка прошлого))
        inputs = self.tokenizer(input_text, return_tensors='pt')
        # inputs.to("mps") p.s. ошибка прошлого))
        generated_tokens = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(generated_tokens[0])
    
# этот класс нужен для предобработки датасета на котором обучаем
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, dataset, tokenizer):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ''
        self.sources = [f"{bos_token}{example[0]}" for example in dataset]
        self.targets = [f"{example[1]}{self.tokenizer.eos_token}" for example in dataset]
        self.IGNORE_INDEX = -100
        self.pad_token_id = 60000
        
    def _tokenize(self, text: str, max_length: int) -> List[int]:
        return self.tokenizer(text,
                              max_length = max_length,
                              truncation = True,
                              add_special_tokens = False)['input_ids']
    
    def __len__(self) -> int:
        return len(self.sources)

    def padding(self, ids: torch.tensor, max_length: int) -> torch.tensor:
        return

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        source_ids = self._tokenize(self.sources[i], self.tokenizer.model_max_length)
        target_ids = self._tokenize(self.targets[i], self.tokenizer.model_max_length - len(source_ids))
        input_ids = torch.tensor(source_ids + target_ids)
        input_pad_len = self.tokenizer.model_max_length - len(input_ids)
        input_ids_pad = torch.tensor(source_ids + target_ids + [self.tokenizer.pad_token_id]*input_pad_len)
        input_att_mask = torch.tensor([[1]*len(input_ids) + [0]*input_pad_len])
        labels = torch.tensor([self.IGNORE_INDEX] * (len(source_ids) - 1) + target_ids + [self.pad_token_id]*(input_pad_len + 1))

        return dict(input_ids=input_ids_pad, input_att_mask=input_att_mask,
                    labels=labels)

        

 
    

def construct_model():
    generation_kwargs = {
        "max_new_tokens": 100,
        "num_beams": 3,
        "early_stopping": True,
        "no_repeat_ngram_size": 2
    }
    model = GPTWrapper()
    model.model.load_state_dict(torch.load('models/gpt_lm/model.pkl'))
    model.model.eval()
    return model, generation_kwargs