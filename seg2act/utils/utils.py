# cut (Chinese version)
def cut(text: str, max_length: int = 60) -> str:
    cutoff_len = int(max_length * 0.8) // 2
    if len(text) > max_length:
        text = f"{text[:cutoff_len]}...{text[-cutoff_len:]}"
    return text

# cut (English version)
def english_cut(text: str, max_length: int = 60) -> str:
    cutoff_len = int(max_length * 0.9) // 2
    if len(text) > max_length:
        words = text.split(' ')
        new_text_left = ''
        while len(new_text_left) < cutoff_len and len(words) > 0:
            new_text_left += words[0] + ' '
            words.pop(0)
        new_text_right = ''
        while len(new_text_right) < cutoff_len and len(words) > 0:
            new_text_right = ' ' + words[-1] + new_text_right
            words.pop(-1)
        text = new_text_left + '...' + new_text_right
    return text

def generate_prompt(datapoint):
    return (
        f"### Tree:\n"
        f"{datapoint['tree']}\n"
        f"### Input:\n"
        f"{datapoint['input']}\n"
        f"### Output:\n"
        f"{datapoint['output']}"
    )