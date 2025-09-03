from vllm.model_executor.models.qwen2 import Qwen2Attention
import torch

def choose_recompute(
        self,###可调用Qwen2Attention类实例种的所有变量
        hidden_states,###当前层输入隐变量，维度为valid_ind中1的个数+prompt的长度，默认头尾prompt均要重算，顺序沿用拼接顺序头prompt+重算位置+尾prompt
        old_k,###预埋的old_kv，维度为不计prompt的全文档长度
        old_v,
        layer_ind,###当前层数
        valid_ind,###重算指示变量，维度为不计prompt的全文档长度，1表示上一层的重算位置
        doc_length,###每个文档长度,用来标记valid_ind的文档分界线,第一个和最后一个是头尾prompt长度
        q,###根据输入hidden_states算出来的qkv，维度顺序与hidden_states一致
        k,
        v,
    ):
        ####Cacheblend示例代码：
        recompute_ratio = 0.25
        begin = doc_length[0]
        end = doc_length[-1]
        num_tokens = len(valid_ind)
        topk_num = int(num_tokens * recompute_ratio)
        if layer_ind == 1:
            if topk_num !=0:
                temp_diff = torch.sum((v[begin:-end]-old_v)**2,dim = 1)
                top_indices = torch.topk(temp_diff, k=topk_num).indices
                for i in range(num_tokens):
                    if valid_ind[i] == 1 and i not in top_indices:
                        valid_ind[i] = 0
            else:
                valid_ind = [0] * num_tokens

        return valid_ind  ##注意输出的valid_ind中的1必须是输入的子集
