import re
from transformers import TextIteratorStreamer, TextStreamer

def check_and_truncate_repetitions(text, threshold=3, min_length=5):
    """
    检测文本中的重复内容并截断
    
    Args:
        text: 要检查的文本
        threshold: 重复阈值，连续出现几次词组算作重复
        min_length: 最小检测重复的词组长度
        
    Returns:
        处理后的文本
    """
    # 添加列表项重复检测
    list_pattern = r'(?:^|\n)\s*[-*•]\s*(.*?)(?=\n\s*[-*•]|\Z)'
    list_items = re.findall(list_pattern, text, re.MULTILINE)
    
    if len(list_items) >= threshold:
        # 检查列表项中的重复
        item_counts = {}
        for item in list_items:
            item_clean = item.strip()
            if len(item_clean) >= min_length:
                if item_clean in item_counts:
                    item_counts[item_clean] += 1
                else:
                    item_counts[item_clean] = 1
        
        for item, count in item_counts.items():
            if count >= threshold:
                # 找到第三次出现的位置并截断
                pattern = f'(?:^|\n)\\s*[-*•]\\s*{re.escape(item)}'
                matches = list(re.finditer(pattern, text, re.MULTILINE))
                if len(matches) >= threshold:
                    # 在第threshold次匹配之前截断
                    truncate_pos = matches[threshold-1].start()
                    return text[:truncate_pos] + "...\n[检测到重复内容，已自动截断]"
    
    # 如果文本很短，直接返回
    if len(text) < min_length * threshold:
        return text
    

    # 方法1: 检测重复的单词或短语 (优先处理"下界合金镐"这样的场景)
    # 使用简单的正则表达式分割文本为单词，包括中文和英文
    word_pattern = re.compile(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\u4e00-\u9fffa-zA-Z0-9\s]+')
    words = word_pattern.findall(text)
    
    # 检测连续重复的单词
    if len(words) > threshold:
        repeat_count = 1
        last_word = words[0]
        
        for i in range(1, len(words)):
            # 跳过标点和空白
            if words[i] in ['、', '，', ',', '。', '.', ' ', '\n', '\t']:
                continue
                
            if words[i] == last_word and len(last_word) >= min_length:
                repeat_count += 1
                if repeat_count >= threshold:
                    # 直接截断在第一个重复的单词之后
                    # 找到第一个重复单词在文本中的位置
                    first_pos = text.find(last_word)
                    if first_pos >= 0:
                        # 确保我们找到的是第一次出现
                        next_pos = first_pos + len(last_word)
                        # 找到适当的截断点(通常在单词后的标点或空格)
                        for punct in ['、', '，', ',', '。', '.', ' ', '；', ';']:
                            punct_pos = text.find(punct, next_pos)
                            if punct_pos > 0 and punct_pos < next_pos + 10:  # 在合理范围内找到标点
                                next_pos = punct_pos + 1
                                break
                        return text[:next_pos] + "..."
            else:
                repeat_count = 1
                last_word = words[i]
    
    # 方法2: 检查特定模式的重复（如"下界合金镐、下界合金镐、下界合金镐"）
    # 直接使用正则表达式查找常见的游戏物品重复模式
    item_pattern = r'([\u4e00-\u9fff]{2,}[镐铲斧剑锹锄锭块矿石]+)[、，,\s]+\1[、，,\s]+\1'
    match = re.search(item_pattern, text)
    if match:
        # 找到重复的物品，截断到第一次出现之后
        item = match.group(1)
        pos = text.find(item)
        if pos >= 0:
            # 找到第一次出现后的合适截断点
            end_pos = pos + len(item)
            for punct in ['、', '，', ',', '。', '.']:
                punct_pos = text.find(punct, end_pos)
                if punct_pos > 0 and punct_pos < end_pos + 5:
                    end_pos = punct_pos + 1
                    break
            return text[:end_pos] + "..."
    
    # 方法3: 检查重复的字符串段落
    for length in range(min_length, min(30, len(text) // threshold)):
        for i in range(len(text) - length * threshold):
            pattern = text[i:i + length]
            is_repeating = True
            
            # 检查是否有连续threshold次重复
            for j in range(1, threshold):
                if text[i + j*length:i + (j+1)*length] != pattern:
                    is_repeating = False
                    break
            
            if is_repeating:
                return text[:i + length] + "..."
    
    # 方法4: 检查更大间隔的重复（不一定是连续的）
    # 特别针对列表中的重复项
    item_list_pattern = re.compile(r'[\u4e00-\u9fff]{2,}[镐铲斧剑锹锄锭块矿石]+')
    items = item_list_pattern.findall(text)
    
    if len(items) >= threshold:
        # 统计每个物品出现的次数
        item_counts = {}
        for item in items:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
        
        # 检查是否有物品重复出现达到阈值
        for item, count in item_counts.items():
            if count >= threshold:
                # 找到第一次出现的位置
                first_pos = text.find(item)
                if first_pos >= 0:
                    # 有时我们可能想保留一些上下文，找到更好的截断点
                    context_end = first_pos + len(item)
                    # 寻找第一次出现后的标点作为截断点
                    for punct in ['、', '，', ',', '。', '.']:
                        punct_pos = text.find(punct, context_end)
                        if punct_pos > 0 and punct_pos < context_end + 15:
                            context_end = punct_pos + 1
                            break
                    return text[:context_end] + "..."
    
    return text


# 创建一个自定义的Streamer类，继承自TextStreamer
class RepetitionCheckingStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=True):
        # 仅传递基本参数给父类，避免参数不匹配的问题
        super().__init__(tokenizer, skip_special_tokens=skip_special_tokens)
        
        # 保存额外的参数
        self.skip_prompt = skip_prompt
        
        # 重复检测相关的状态
        self.generated_text = ""
        self.last_printed_text = ""
        self.threshold = 3
        self.min_length = 3
        self.found_repetition = False
    
    def reset(self):
        """重置流媒体状态，在每次新生成开始前调用"""
        self.generated_text = ""
        self.last_printed_text = ""
        self.found_repetition = False
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """处理模型生成的完成文本片段"""
        # 移除特殊标识符
        text = text.replace("<|im_end|>", "")
        
        # 将新文本添加到已生成的文本中
        self.generated_text += text
        
        # 如果已经检测到重复，不再继续处理
        if self.found_repetition:
            return
        
        # 检查重复模式
        cleaned_text = check_and_truncate_repetitions(self.generated_text, self.threshold, self.min_length)
        
        # 如果找到重复并截断了文本
        if len(cleaned_text) < len(self.generated_text):
            # 计算新增部分（从上次打印位置到截断位置）
            new_part = cleaned_text[len(self.last_printed_text):]
            if new_part:
                print(new_part, end="")
            
            # 输出截断提示
            print("\n[检测到重复内容，已自动截断]", end="")
            
            # 更新状态
            self.found_repetition = True
            self.last_printed_text = cleaned_text
            self.generated_text = cleaned_text  # 更新为截断后的文本
        else:
            # 正常打印新文本
            print(text, end="", flush=True)
            self.last_printed_text = self.generated_text
    
    def get_final_text(self):
        """返回最终的文本(可能被截断)"""
        # 确保返回的文本中不包含特殊标识符
        return self.generated_text.replace("<|im_end|>", "")



# 创建一个自定义的Streamer类，继承自TextIteratorStreamer
class RepetitionIteratorStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        # 初始化父类
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        
        # 重复检测相关的状态
        self.generated_text = ""
        self.last_printed_text = ""
        self.threshold = 5
        self.min_length = 5
        self.found_repetition = False
    
    def reset(self):
        """重置流媒体状态，在每次新生成开始前调用"""
        super().reset()  # 调用父类的reset方法
        self.generated_text = ""
        self.last_printed_text = ""
        self.found_repetition = False
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # 移除特殊标识符
        text = text.replace("<|im_end|>", "")
        
        # 如果已检测到重复，不再添加新内容
        if self.found_repetition:
            # 不添加新内容，也不调用父类方法传递文本
            return
        
        # 将新文本添加到已生成的文本中
        self.generated_text += text
        
        # 检查重复模式
        cleaned_text = check_and_truncate_repetitions(self.generated_text, self.threshold, self.min_length)
        
        # 如果找到重复并截断了文本
        if len(cleaned_text) < len(self.generated_text):
            self.found_repetition = True
            # 添加明确的截断标记
            if not cleaned_text.endswith("[检测到重复内容，已自动截断]"):
                self.generated_text = cleaned_text + "...\n[检测到重复内容，已自动截断]"
            else:
                self.generated_text = cleaned_text
                
            # 调用父类方法，但只传递额外的截断标记部分
            diff = self.generated_text[-(len(self.generated_text) - len(self.last_printed_text)):]
            super().on_finalized_text(diff, True)  # 将stream_end设为True以表示生成应该结束
        else:
            # 正常传递文本
            super().on_finalized_text(text, stream_end)
            
        self.last_printed_text = self.generated_text
        
    def get_final_text(self):
        """返回最终的文本(可能被截断)"""
        # 确保返回的文本中不包含特殊标识符
        return self.generated_text.replace("<|im_end|>", "")