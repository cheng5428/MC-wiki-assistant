import json
import os
import random

def extract_text_conversations1(input_file, output_file, total_num=1000, min_words=50):
    """
    从新格式的JSON文件中提取纯文本对话，并保存到TXT文件中
    JSON格式: [{"instruction": "...", "input": "...", "output": "..."}, ...]
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出TXT文件路径
        total_num (int): 需要提取的对话数量，None表示全部提取
        min_words (int): 对话内容的最小字数（包括问题和回答）
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 随机打乱数据（如果需要随机抽取）
        if total_num is not None:
            random.shuffle(data)
        
        # 筛选有效的对话（字数达标的对话）
        valid_items = []
        for item in data:
            # 确保所有字段都是字符串类型
            instruction = str(item.get("instruction", ""))
            input_text = str(item.get("input", ""))
            output_text = str(item.get("output", "")).replace("\n", " ").strip()
            
            if not instruction or not output_text:
                continue
            # if not instruction.endswith("?"):
            #     continue

            # 计算总字数
            total_words = len(instruction) + len(input_text) + len(output_text)
            
            if total_words >= min_words:
                valid_items.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
        
        print(f"总共找到 {len(valid_items)} 个有效对话")
        
        # 控制抽取的数量
        if total_num is not None:
            sample_count = min(total_num, len(valid_items))
            valid_items = valid_items[:sample_count]
        
        # 打开输出文件准备写入
        with open(output_file, 'a+', encoding='utf-8') as f:
            for item in valid_items:
                # 提取指令、输入和输出
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                
                # 构建用户问题文本
                user_question = instruction 
                if input_text:
                    user_question += input_text
                
                # 写入问题和回答，并在对话间添加空行
                f.write(user_question + "\n")
                f.write(output_text + "\n\n")

        print(f"转换完成！成功提取并写入 {len(valid_items)} 条对话到 {output_file}")
        
    except Exception as e:
        print(f"提取过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def extract_text_conversations2(input_file, output_file, total_num=1000, min_words=50):
    """
    从新格式的JSON文件中提取纯文本对话，并保存到TXT文件中
    JSON格式: [{"question": "...", "answer": "...", "source": "..."}, ...]
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出TXT文件路径
        total_num (int): 需要提取的对话数量，None表示全部提取
        min_words (int): 对话内容的最小字数（包括问题和回答）
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 随机打乱数据（如果需要随机抽取）
        if total_num is not None:
            random.shuffle(data)
        
        # 筛选有效的对话（字数达标的对话）
        valid_items = []
        for item in data:
            # 确保所有字段都是字符串类型
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).replace("\n", " ").strip()
            
            if not question or not answer:
                continue
            if not question.endswith("?"):
                continue
            # 计算总字数
            total_words = len(question) + len(answer)
            
            if total_words >= min_words and question and answer:
                valid_items.append({
                    "question": question,
                    "answer": answer
                })
        
        print(f"总共找到 {len(valid_items)} 个有效对话")
        
        # 控制抽取的数量
        if total_num is not None:
            sample_count = min(total_num, len(valid_items))
            valid_items = valid_items[:sample_count]
        
        # 打开输出文件准备写入
        with open(output_file, 'a+', encoding='utf-8') as f:
            for item in valid_items:
                # 提取问题和答案
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                # 写入问题和回答，并在对话间添加空行
                f.write(question + "\n")
                f.write(answer + "\n\n")

        print(f"转换完成！成功提取并写入 {len(valid_items)} 条对话到 {output_file}")
        
    except Exception as e:
        print(f"提取过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def extract_text_conversations3(input_file, output_file, total_num=1000, min_words=50):
    """
    从新的对话格式JSON文件中提取纯文本对话，并保存到TXT文件中
    JSON格式: {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出TXT文件路径
        total_num (int): 需要提取的对话数量，None表示全部提取
        min_words (int): 对话内容的最小字数（包括问题和回答）
    """
    try:
        # 读取输入文件
        conversations_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            # 逐行读取JSON对象（JSONL格式）
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        conversations_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"跳过无效的JSON行: {line[:100]}... 错误: {e}")
                        continue
        
        # 如果没有数据，尝试作为单个JSON文件读取
        if not conversations_data:
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        conversations_data = data
                    else:
                        conversations_data = [data]
                except json.JSONDecodeError as e:
                    print(f"无法解析JSON文件: {e}")
                    return
        
        print(f"成功读取 {len(conversations_data)} 个对话记录")
        
        # 随机打乱数据（如果需要随机抽取）
        if total_num is not None:
            random.shuffle(conversations_data)
        
        # 筛选有效的对话（字数达标的对话）
        valid_items = []
        for item in conversations_data:
            try:
                # 获取对话列表
                conversations = item.get("conversations", [])
                
                if len(conversations) < 2:
                    continue
                
                # 查找用户问题和助手回答
                user_content = ""
                assistant_content = ""
                
                for conv in conversations:
                    role = conv.get("role", "").lower()
                    content = str(conv.get("content", "")).strip()
                    
                    if role == "user" and not user_content:
                        user_content = content
                    elif role == "assistant" and not assistant_content:
                        assistant_content = content
                
                # 检查是否找到有效的问答对
                if not user_content or not assistant_content:
                    continue
                
                # 清理内容：移除换行符并处理格式
                user_content = user_content.replace("\n", " ").strip()
                assistant_content = assistant_content.replace("\n", " ").strip()
                
                # 确保问题以问号结尾（可选，根据需要调整）
                if not user_content.endswith(("?", "？")):
                    user_content += "?"
                
                # 计算总字数
                total_words = len(user_content) + len(assistant_content)
                
                if total_words >= min_words:
                    valid_items.append({
                        "question": user_content,
                        "answer": assistant_content
                    })
                    
            except Exception as e:
                print(f"处理对话时出错: {e}")
                continue
        
        print(f"总共找到 {len(valid_items)} 个有效对话")
        
        # 控制抽取的数量
        if total_num is not None:
            sample_count = min(total_num, len(valid_items))
            valid_items = valid_items[:sample_count]
            print(f"按要求提取前 {sample_count} 个对话")
        
        # 打开输出文件准备写入
        with open(output_file, 'w', encoding='utf-8') as f:  # 使用'w'模式覆盖文件
            for i, item in enumerate(valid_items):
                # 提取问题和答案
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                # 写入问题和回答，并在对话间添加空行
                f.write(question + "\n")
                f.write(answer + "\n")
                
                # 除了最后一个对话，都添加空行分隔
                if i < len(valid_items) - 1:
                    f.write("\n")

        print(f"转换完成！成功提取并写入 {len(valid_items)} 条对话到 {output_file}")
        
    except Exception as e:
        print(f"提取过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def convert_txt_to_json(input_file, output_file):
    """
    将txt文件转换为json格式，处理严格的一行问题一行答案格式
    
    参数:
    input_file (str): 输入txt文件的路径
    output_file (str): 输出json文件的路径
    
    txt文件格式:
    问题1?
    答案1
    
    问题2?
    答案2
    
    ...
    """
    import json
    
    # 读取txt文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 按空行分割内容获取每个问答对
    pairs = content.split("\n\n")
    
    # 初始化结果列表
    result = []
    skipped = 0
    cnt = 0
    with open(output_file, 'a+', encoding='utf-8') as f:
        # 处理每个问答对
        for pair in pairs:
            # 跳过空内容
            if not pair.strip():
                continue
            # 分割问题和答案
            lines = pair.strip().split("\n")
            
            # 检查是否是有效的问答对（至少两行）
            if len(lines) < 2:
                print(f"警告: 跳过不完整的问答对: {pair}")
                skipped += 1
                continue
            
            question = lines[0].strip()
            answer = lines[1].strip()
            
            # 添加到结果中
            result.append({"role": "user", "content": question})
            result.append({"role": "assistant", "content": answer})
        
            # 将结果写入json文件
            if len(result) == 2:
                cnt += 1
                json.dump({"conversations": result}, f, ensure_ascii=False, indent=None, separators=(',', ':'))
                f.write('\n')  # 每个 JSON 对象单独一行
                result = []
    
    print(f"转换完成! 已将 {cnt} 条问答对写入 {output_file}")
    if skipped > 0:
        print(f"注意: 跳过了 {skipped} 个不完整的问答对")
    
    return result

def clean_repetitive_content(input_file, output_file, min_repeat=10, check_length=3):
    """
    清除文件中包含异常重复内容的问答对
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        min_repeat (int): 判定为异常的最小重复次数
        check_length (int): 需要检查的字符/字符串长度，范围1-5
    
    Returns:
        tuple: (保留的问答对数量, 删除的问答对数量)
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 按空行分割获取问答对
    pairs = content.split("\n\n")
    
    cleaned_pairs = []
    removed_count = 0
    
    def has_repetitive_patterns(text):
        """检查文本中是否有异常的重复模式"""
        if not text:
            return False
            
        # 检查不同长度的重复模式
        for pattern_len in range(1, min(6, check_length+1)):
            # 遍历文本，查找重复模式
            for i in range(len(text) - pattern_len * min_repeat):
                pattern = text[i:i+pattern_len]
                # 跳过空格和标点符号的模式
                if pattern.isspace() or pattern in ',.?!;:\'\"':
                    continue
                    
                # 计算连续重复次数
                repeat_count = 1
                pos = i + pattern_len
                while pos <= len(text) - pattern_len and text[pos:pos+pattern_len] == pattern:
                    repeat_count += 1
                    pos += pattern_len
                    
                # 如果重复次数超过阈值，判定为异常
                if repeat_count >= min_repeat:
                    return True
                    
        return False
    
    # 处理每个问答对
    for pair in pairs:
        # 跳过空内容
        if not pair.strip():
            continue
            
        # 分割问题和回答
        lines = pair.strip().split("\n")
        
        # 检查是否是有效的问答对
        if len(lines) < 2:
            continue
            
        question = lines[0].strip()
        answer = lines[1].strip()
        
        # 检查问题和回答中是否有异常重复
        if has_repetitive_patterns(question) or has_repetitive_patterns(answer):
            removed_count += 1
            continue
        
        # 保留正常的问答对
        cleaned_pairs.append(f"{question}\n{answer}")
    
    # 写入输出文件
    with open(output_file, 'a+', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned_pairs))
    
    print(f"清理完成！从 {len(pairs)} 个问答对中移除了 {removed_count} 个异常问答对")
    print(f"保留了 {len(cleaned_pairs)} 个有效问答对，已写入到 {output_file}")
    
    return (len(cleaned_pairs), removed_count)

def select_longest_conversations_with_stats(input_file, output_file, ratio=0.1):
    """
    按比例筛选出txt文件中最长的对话，并显示详细统计信息
    
    Args:
        input_file (str): 输入txt文件路径
        output_file (str): 输出txt文件路径
        ratio (float): 筛选比例，0.1表示选择最长的10%对话
    
    Returns:
        dict: 包含统计信息的字典
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 按空行分割获取问答对
        pairs = content.split("\n\n")
        
        # 过滤掉空内容并计算长度
        valid_pairs = []
        lengths = []
        
        for pair in pairs:
            if pair.strip():
                lines = pair.strip().split("\n")
                if len(lines) >= 2:
                    question = lines[0].strip()
                    answer = lines[1].strip()
                    total_length = len(question) + len(answer)
                    valid_pairs.append((total_length, pair.strip()))
                    lengths.append(total_length)
        
        if not valid_pairs:
            print("错误: 没有找到有效的问答对")
            return {}
        
        # 按长度降序排序
        valid_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # 计算需要选择的对话数量
        total_count = len(valid_pairs)
        select_count = max(1, int(total_count * ratio))
        
        # 选择最长的对话
        selected_pairs = [pair_content for _, pair_content in valid_pairs[:select_count]]
        selected_lengths = [length for length, _ in valid_pairs[:select_count]]
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(selected_pairs))
        
        # 计算统计信息
        import statistics
        stats = {
            'total_count': total_count,
            'selected_count': select_count,
            'selection_ratio': ratio,
            'max_length': max(lengths),
            'min_length': min(lengths),
            'avg_length': statistics.mean(lengths),
            'median_length': statistics.median(lengths),
            'selected_max_length': max(selected_lengths),
            'selected_min_length': min(selected_lengths),
            'selected_avg_length': statistics.mean(selected_lengths),
            'selected_median_length': statistics.median(selected_lengths)
        }
        
        # 打印统计信息
        print(f"\n=== 对话筛选统计 ===")
        print(f"总对话数量: {stats['total_count']}")
        print(f"选择数量: {stats['selected_count']}")
        print(f"筛选比例: {stats['selection_ratio']:.1%}")
        print(f"\n=== 原始数据统计 ===")
        print(f"最长对话: {stats['max_length']} 字符")
        print(f"最短对话: {stats['min_length']} 字符")
        print(f"平均长度: {stats['avg_length']:.1f} 字符")
        print(f"中位数长度: {stats['median_length']:.1f} 字符")
        print(f"\n=== 筛选结果统计 ===")
        print(f"选中最长: {stats['selected_max_length']} 字符")
        print(f"选中最短: {stats['selected_min_length']} 字符")
        print(f"选中平均: {stats['selected_avg_length']:.1f} 字符")
        print(f"选中中位数: {stats['selected_median_length']:.1f} 字符")
        print(f"\n结果已保存到: {output_file}")
        
        return stats
        
    except Exception as e:
        print(f"筛选过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
def split_conversation_file(input_file, num_splits=4, output_prefix="split_"):
    """
    将包含问答对的txt文件平均切分到多个文件中
    
    Args:
        input_file (str): 输入txt文件路径
        num_splits (int): 切分的文件数量
        output_prefix (str): 输出文件的前缀名称
    
    Returns:
        tuple: (切分成功的文件数量, 每个文件的问答对数量列表)
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 按空行分割获取问答对
        pairs = content.split("\n\n")
        
        # 过滤掉空内容
        valid_pairs = []
        for pair in pairs:
            if pair.strip():
                valid_pairs.append(pair.strip())
        
        if not valid_pairs:
            print("错误: 没有找到有效的问答对")
            return (0, [])
        
        total_pairs = len(valid_pairs)
        random.shuffle(valid_pairs)

        print(f"总共找到 {total_pairs} 个问答对")
        

        # 计算每个文件应该包含的问答对数量
        pairs_per_file = total_pairs // num_splits
        remainder = total_pairs % num_splits
        
        # 分配问答对到各个文件
        file_counts = []
        start_idx = 0
        
        for i in range(num_splits):
            # 前 remainder 个文件多分配一个问答对
            current_count = pairs_per_file + (1 if i < remainder else 0)
            end_idx = start_idx + current_count
            
            # 获取当前文件的问答对
            current_pairs = valid_pairs[start_idx:end_idx]
            
            # 生成输出文件名
            output_file = f"{output_prefix}{i+1}.txt"
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(current_pairs))
            
            file_counts.append(len(current_pairs))
            print(f"文件 {output_file}: {len(current_pairs)} 个问答对")
            
            start_idx = end_idx
        
        print(f"\n切分完成！将 {total_pairs} 个问答对分配到 {num_splits} 个文件中")
        print(f"平均每个文件: {pairs_per_file} 个问答对")
        if remainder > 0:
            print(f"前 {remainder} 个文件各多分配 1 个问答对")
        
        return (num_splits, file_counts)
        
    except Exception as e:
        print(f"切分过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return (0, [])

def merge_json_files_with_ratio(task_file, identity_file, output_file, ratio=1/5, question_mark_removal_prob=0.4, task_removal_prob=0.0):
    """
    合并任务数据和自我认知数据，按照指定比例采样，必要时重复自我认知数据
    随机去除数据集中问题的问号（包括自我认知和任务数据）
    按照设定的概率随机去除部分任务数据
    
    Args:
        task_file (str): 任务数据的JSON文件路径
        identity_file (str): 自我认知数据的JSON文件路径
        output_file (str): 输出合并结果的JSON文件路径
        ratio (float): 自我认知数据与任务数据的比例，默认1:5
        question_mark_removal_prob (float): 去除问号的概率，默认0.4
        task_removal_prob (float): 随机去除任务数据的概率，默认0.0（不去除）
    
    Returns:
        dict: 包含合并统计信息的字典
    """
    import json
    import random
    
    def remove_question_mark(item, removal_prob):
        """
        根据概率随机去除数据中问题的问号
        
        Args:
            item (dict): 数据项
            removal_prob (float): 去除问号的概率
            
        Returns:
            dict: 可能被修改的数据项
        """
        # 创建一个数据副本，避免修改原始数据
        modified_item = json.loads(json.dumps(item))
        
        if random.random() < removal_prob:
            # 处理conversations格式
            if "conversations" in modified_item:
                for conv in modified_item["conversations"]:
                    if conv.get("role") == "user" and isinstance(conv.get("content"), str):
                        content = conv["content"]
                        if content.endswith("?") or content.endswith("？"):
                            conv["content"] = content[:-1].strip()
        
        return modified_item
    
    try:
        # 读取任务数据
        raw_task_data = []
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        raw_task_data.append(data)
                    except json.JSONDecodeError:
                        continue
        
        # 根据概率随机去除部分任务数据
        original_task_count = len(raw_task_data)
        if task_removal_prob > 0:
            task_data = []
            for item in raw_task_data:
                if random.random() >= task_removal_prob:  # 保留该数据项
                    # 随机去除任务数据中的问号
                    modified_data = remove_question_mark(item, question_mark_removal_prob)
                    task_data.append(modified_data)
        else:
            # 不需要去除任务数据，只处理问号
            task_data = [remove_question_mark(item, question_mark_removal_prob) for item in raw_task_data]
        
        # 读取自我认知数据
        identity_data = []
        with open(identity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        identity_data.append(data)
                    except json.JSONDecodeError:
                        continue
        
        print(f"读取任务数据: {original_task_count} 条")
        if task_removal_prob > 0:
            print(f"随机去除后任务数据: {len(task_data)} 条 (去除率: {task_removal_prob:.1%})")
        print(f"读取自我认知数据: {len(identity_data)} 条")
        
        # 根据任务数据和比例计算需要的自我认知数据数量
        required_identity_count = int(len(task_data) * ratio)
        
        processed_identity_data = []
        
        # 如果自我认知数据不足，通过重复补充
        if len(identity_data) < required_identity_count:
            # 计算需要重复的次数
            repeat_times = required_identity_count // len(identity_data)
            remainder = required_identity_count % len(identity_data)
            
            # 重复自我认知数据，并随机去除部分问号
            for _ in range(repeat_times):
                for item in identity_data:
                    modified_item = remove_question_mark(item, question_mark_removal_prob)
                    processed_identity_data.append(modified_item)
            
            # 添加余下需要的数据
            for item in random.sample(identity_data, remainder):
                modified_item = remove_question_mark(item, question_mark_removal_prob)
                processed_identity_data.append(modified_item)
            
            print(f"自我认知数据不足，通过重复扩充至 {len(processed_identity_data)} 条")
            print(f"重复次数: {repeat_times} 次，额外采样: {remainder} 条")
        else:
            # 如果自我认知数据充足，随机采样所需数量
            for item in random.sample(identity_data, required_identity_count):
                modified_item = remove_question_mark(item, question_mark_removal_prob)
                processed_identity_data.append(modified_item)
            
            print(f"自我认知数据充足，随机采样 {required_identity_count} 条")
        
        # 统计问号被移除的数量
        removed_marks_task = 0
        removed_marks_identity = 0
        
        # 检查任务数据中问号被移除的情况
        for item in task_data:
            if "conversations" in item:
                for conv in item["conversations"]:
                    if conv.get("role") == "user" and isinstance(conv.get("content"), str):
                        content = conv["content"]
                        if not (content.endswith("?") or content.endswith("？")):
                            # 粗略统计，可能不完全准确
                            removed_marks_task += 1
        
        # 检查自我认知数据中问号被移除的情况
        for item in processed_identity_data:
            if "conversations" in item:
                for conv in item["conversations"]:
                    if conv.get("role") == "user" and isinstance(conv.get("content"), str):
                        content = conv["content"]
                        if not (content.endswith("?") or content.endswith("？")):
                            removed_marks_identity += 1
        
        # 合并数据
        merged_data = task_data + processed_identity_data
        random.shuffle(merged_data)  # 打乱合并后的数据顺序
        
        print(f"合并后总数据量: {len(merged_data)} 条")
        print(f"其中任务数据: {len(task_data)} 条 ({len(task_data)/len(merged_data):.2%})")
        print(f"其中自我认知数据: {len(processed_identity_data)} 条 ({len(processed_identity_data)/len(merged_data):.2%})")
        print(f"随机去除问号的概率: {question_mark_removal_prob:.2%}")
        print(f"任务数据中问号被移除的估计条数: {removed_marks_task} 条")
        print(f"自我认知数据中问号被移除的估计条数: {removed_marks_identity} 条")
        
        # 写入合并后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in merged_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"成功将合并数据写入 {output_file}")
        
        # 返回统计信息
        stats = {
            'original_task_count': original_task_count,
            'filtered_task_count': len(task_data),
            'task_removal_rate': (original_task_count - len(task_data)) / original_task_count if original_task_count > 0 else 0,
            'total_items': len(merged_data),
            'task_items': len(task_data),
            'identity_items': len(processed_identity_data),
            'task_ratio': len(task_data) / len(merged_data),
            'identity_ratio': len(processed_identity_data) / len(merged_data),
            'required_identity': required_identity_count,
            'removed_question_marks_task': removed_marks_task,
            'removed_question_marks_identity': removed_marks_identity,
            'question_mark_removal_prob': question_mark_removal_prob
        }
        
        return stats
        
    except Exception as e:
        print(f"合并过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {}

# 使用示例
if __name__ == "__main__":
    # 如果你想单独运行这个函数
    # 合并任务数据和自我认知数据
    task_file = "mc_data_conversation.json"  # 替换为你的任务数据文件路径
    identity_file = "identity.json"  # 替换为你的自我认知数据文件路径
    output_file = "merged_data.json"  # 合并后的输出文件路径
    
    # 按1:5比例合并数据（自我认知:任务 = 1:5）
    merge_json_files_with_ratio(task_file, identity_file, output_file, ratio=1/5, question_mark_removal_prob=0.45, task_removal_prob=0.25)

