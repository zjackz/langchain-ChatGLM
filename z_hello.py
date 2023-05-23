





from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/models/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("/hy-tmp/models/chatglm-6b-int4",trust_remote_code=True).half().cuda()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)




# import torch


# print (torch.cuda.is_available())


