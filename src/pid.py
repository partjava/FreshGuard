"""
位置式 PID 控制器（参考 Yahboom 官方实现）
用于平滑控制小车转向和速度
"""


class PositionalPID:
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.SystemOutput = 0.0
        self.ResultValueBack = 0.0
        self.PidOutput = 0.0
        self.PIDErrADD = 0.0
        self.ErrBack = 0.0

    def SetStepSignal(self, StepSignal):
        Err = StepSignal - self.SystemOutput
        self.PidOutput = self.Kp * Err + self.Ki * self.PIDErrADD + self.Kd * (Err - self.ErrBack)
        self.PIDErrADD += Err
        self.ErrBack = Err

    def SetInertiaTime(self, InertiaTime, SampleTime):
        self.SystemOutput = (InertiaTime * self.ResultValueBack +
                             SampleTime * self.PidOutput) / (SampleTime + InertiaTime)
        self.ResultValueBack = self.SystemOutput
