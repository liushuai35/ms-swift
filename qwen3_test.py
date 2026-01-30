import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载模型
model_path = "/home/shuai.liu01/merged_qwen3_4b_with_special_token"

# 加载 Tokenizer (加上 trust_remote_code=True 以防万一)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True
).to("cuda")

# 2. 准备数据
# 核心技巧：System Prompt 显式禁止工具，并强调纯文本
# 定义一个简短的范例（让模型照猫画虎）
example_input = "产品A的销量在Q1增长了10%。\n下一季度预计持平。\n表1. 产品A详细参数\n型号 尺寸 重量\nA1 10mm 5g"
example_output = "产品A的销量在Q1增长了10%。\n下一季度预计持平。\n\n===SEGMENT===\n\n表1. 产品A详细参数\n型号 尺寸 重量\nA1 10mm 5g"

real_input = """2.1 数据保密协议
所有员工入职时必须签署NDA。关于客户数据、专有算法和财务记录的信息被视为绝密。任何违规行为将导致立即终止合同
并承担法律责任。请参考员工手册第4章获取更多细节。性能测试结果分析。如图4所示，在应用补丁后，系统吞吐量趋于稳定。平均响应时间从200ms下降到150ms。然而，内存使用率在高峰时段仍然偏高。建议在下个版本中针对数据库连接池进行优化。表 5. 2023年第四季度销售数据汇总；区域      销售额 (万)    同比增长华东      450           +12%华南      320           -5%华北      280           +8% 注：数据截止至12月31日。"""

messages = [
    # 1. System Prompt
    {"role": "system", "content": "你是一个RAG文档助手。任务：在语义独立的内容块之间插入“===SEGMENT===”标记。保持原文一致。"},
    
    # # 2. One-Shot Example (核心修复点)
    {"role": "user", "content": example_input},
    {"role": "assistant", "content": example_output},
    
    # 3. Real Input
    {"role": "user", "content": real_input}
]

# 3. 应用模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 4. 编码 (自动生成 attention_mask)
model_inputs = tokenizer(text, return_tensors="pt").to("cuda")

# ----------------------------------------------------------------------------
# 【核心修复技巧】构造 bad_words_ids，强制禁止模型生成工具标签
# 这能从根本上解决 </tool_call> 导致的乱码问题
# ----------------------------------------------------------------------------
forbidden_tokens = ["<tool_call>", "</tool_call>", "<|tool_call|>", "</|tool_call|>"]
bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in forbidden_tokens]

# 5. 生成 (Greedy Search)
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,           # 修复：传入 **字典，自动包含 attention_mask
        max_new_tokens=4096,
        do_sample=False,          # 贪婪搜索
        temperature=None,
        top_p=None,
        bad_words_ids=bad_words_ids, # 强行禁止工具标签
        pad_token_id=tokenizer.eos_token_id
    )

# 6. 切片与解码
input_len = model_inputs.input_ids.shape[1]
generated_ids = [output_ids[input_len:] for output_ids in generated_ids]
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("=" * 20 + " 最终结果 " + "=" * 20)
print(result)