from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = "/home/shuai.liu01/merged_qwen3_4b_with_special_token"
model_path = "/home/shuai.liu01/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# 构造输入 prompt
system_prompt = "按语义分段：请在内容独立处插入换行及“===SEGMENT===”标记。保持原文完全一致，禁止删改。"
input_text = """2.1 数据保密协议
所有员工入职时必须签署NDA。关于客户数据、专有算法和财务记录的信息被视为绝密。任何违规行为将导致立即终止合同并承担法律责任。请参考员工手册第4章获取更多细节。

性能测试结果分析
如图4所示，在应用补丁后，系统吞吐量趋于稳定。平均响应时间从200ms下降到150ms。然而，内存使用率在高峰时段仍然偏高。建议在下个版本中针对数据库连接池进行优化。

表 5. 2023年第四季度销售数据汇总
区域      销售额 (万)    同比增长
华东      450           +12%
华南      320           -5%
华北      280           +8%
注：数据截止至12月31日。"""
prompt = f"### system\n{system_prompt}\n### input\n{input_text}\n### output\n"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成配置
outputs = model.generate(
    **inputs,
    temperature=0.2,
    top_k=1,
    top_p=0.9,
    do_sample=False,
    max_new_tokens=1024,
    pad_token_id=tokenizer.eos_token_id
)

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result)