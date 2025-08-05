import gtsam, time, yaml
from sota.factor_graph.factors import DeepImuFactor, RangeFactorWithBias
from sota.imu.repiln.infer_repiln import DeepImuFrontend
from sota.uwb.infer_uwb import UwbML
from miluv.data import DataLoader

class FGCoordinator:
    def __init__(self, exp_name):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values= gtsam.Values()
        self.isam  = gtsam.ISAM2()
        self.di    = DeepImuFrontend()
        self.uwb   = UwbML(exp_name)
        self.mv    = DataLoader(exp_name, cir=True, height=True)
        # initial priors …
    def run(self):
        while True:
            # 1) Pop next timestamp from MILUV dataloader iterator
            # 2) If IMU – accumulate -> DeepImuFactor (every 200 samps)
            # 3) If UWB range – add RangeFactorWithBias
            # 4) If height – add simple unary
            # 5) Update iSAM2, retrieve latest pose for control / log
            self.isam.update(self.graph, self.values)
            time.sleep(0.005)
