class StatefulDetector(object):
    # 实现一个简单的状态检测器，根据输入的对数M和阈值，判断是否检测到异常。
    # 状态值S的更新采用了漏出因子，使得较早的的异常信息逐渐衰减
    def __init__(self, sigma, tau):
        self.S = 0
        self.sigma = sigma
        self.tau = tau

    def __call__(self, M):
        self.S = max(0.0, self.S + M - self.sigma)
        if self.S > self.tau:
            temp = self.S
            self.S = 0
        #     return temp, True
        # else:
        #     return self.S, False
            return temp, 1
        else:
            return self.S, 0

class StatelessDetector(object):
    def __init__(self, tau):
        self.tau = tau

    def __call__(self, M):
        if M > self.tau:
            return True
        else:
            return False