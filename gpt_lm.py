import transformers
import torch
from torch import nn
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
    
    def train_n_save(model, train_texts, learning_rate: float = 3e-4, N_ITERATIONS: int = 100):
        device = 'mps'
        model.model.to(device)

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.model.parameters(), lr= 0.0000001)
        cur_iteration = 0
        loss_values = []

        train_batch = SupervisedDataset(
            dataset=train_texts[['prompt', 'original_response']].values,
            tokenizer=model.tokenizer,
        )

        train_batch_loader = DataLoader(
            dataset=train_batch,
            batch_size=1,
            shuffle=False,
        )

        for x in train_batch_loader:
            if cur_iteration == N_ITERATIONS:
                break
            model.model.train()
            input_tokens = x['input_ids'].to(device)
            attention_mask = x['input_att_mask'].to(device)
            labels = x['labels'].clone().to(device)
            out_logits = model.model(input_ids=input_tokens, attention_mask=attention_mask).logits
            labels[labels == 60000] = -100

            loss_value = loss(out_logits.permute(0, 2, 1), labels)
            print(f"Loss value: {loss_value.item()}, Iteration: {cur_iteration}")
            loss_value.backward()
            optimizer.step()

            model.model.eval()
            cur_iteration += 1
            loss_values.append(loss_value.cpu().detach().numpy())
        torch.save(model.model.state_dict(), 'models/gpt_lm/model.pkl')
        model.model.to('cpu')

        return loss_values

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