
class SimpleStreamer(Streamer):
    def __init__(self, end_token: int, voc_size: int):
        self.end_token = end_token
        self.voc_size = voc_size
        self.n = 0
        self.tokens = []
    
    def put(self, token: Tensor) -> tuple[bool, Tensor, list[int]]:
        L, = token.shape
        self.tokens += token.tolist()
        is_end = token[-1] == self.end_token # prompt中にend_tokenがある場合も考慮している。
        position = torch.arange(self.n, self.n+L, device=token.device)
        next_token_range = list(range(self.voc_size))
        return is_end, position, next_token_range
