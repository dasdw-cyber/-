def calculate_y(x_values):
    """
    根据公式 y = 1.5334x - 25.382 计算结果
    支持单个数值或数值列表
    """
    slope = 1.5334
    intercept = -25.382

    # 如果输入是列表或元组，进行批量计算
    if isinstance(x_values, (list, tuple)):
        results = [round(slope * x + intercept, 4) for x in x_values]
        return results
    # 如果是单个数值
    else:
        result = slope * x_values + intercept
        return round(result, 4)


# --- 使用示例 ---

# 1. 计算单个数字
input_val = 46.56
output = calculate_y(input_val)
print(f"单个计算结果: 当 x={input_val} 时, y={output}")

# 2. 批量计算（你可以把刚才图片里的数据直接放进这里）
data_list = [50.3, 47.5, 47.7, 49.05]
batch_output = calculate_y(data_list)
print(f"批量计算结果: {batch_output}")

# 3. 交互式输入示例
try:
    user_input = input("\n请输入数字（多个数字请用空格或逗号隔开）: ")
    # 处理字符串输入并转换为浮点数列表
    clean_input = [float(i) for i in user_input.replace(',', ' ').split()]

    if len(clean_input) == 1:
        print(f"计算结果: {calculate_y(clean_input[0])}")
    else:
        print(f"批量计算结果: {calculate_y(clean_input)}")
except ValueError:
    print("输入错误，请输入有效的数字。")