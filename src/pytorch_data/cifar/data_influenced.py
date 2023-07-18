import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

from .data import CIFAR100Data

__all__lp = [
    "CIFAR100HighInfData",
]


# high influence subset, random subset
# from https://pluskid.github.io/influence-memorization/
HIGH_INFLUENCE_CIFAR100 = np.array([
    50, 215, 259, 273, 392, 587, 605, 651, 673, 768, 855, 865, 867, 1034, 1243,
    1300, 1351, 1358, 1394, 1443, 1513, 1601, 1627, 1653, 1664, 1670, 1689,
    1690, 1785, 1796, 1913, 2056, 2083, 2180, 2222, 2263, 2303, 2312, 2385,
    2575, 2594, 2617, 2650, 2759, 2861, 2879, 3000, 3103, 3137, 3263, 3362,
    3429, 3571, 3630, 3667, 3729, 3774, 3777, 3780, 3837, 3849, 3861, 3906,
    3980, 4024, 4076, 4131, 4208, 4259, 4277, 4282, 4410, 4551, 4664, 4688,
    4755, 4800, 4805, 4935, 4962, 4969, 5016, 5034, 5055, 5072, 5100, 5142,
    5167, 5203, 5259, 5265, 5402, 5423, 5445, 5469, 5511, 5556, 5568, 5598,
    5676, 5754, 5807, 5855, 5949, 5963, 6007, 6141, 6168, 6188, 6204, 6216,
    6233, 6303, 6317, 6339, 6349, 6468, 6571, 6602, 6625, 6639, 6666, 6810,
    6835, 6857, 6965, 6972, 7019, 7050, 7110, 7126, 7133, 7216, 7402, 7484,
    7519, 7523, 7578, 7599, 7606, 7627, 7636, 7653, 7780, 7816, 7863, 7908,
    7973, 7985, 8019, 8125, 8139, 8203, 8228, 8239, 8265, 8333, 8358, 8728,
    8749, 8762, 8789, 8808, 9072, 9147, 9280, 9389, 9464, 9501, 9505, 9525,
    9645, 9716, 9767, 9772, 9860, 9866, 9916, 9920, 9964, 9987, 10042, 10055,
    10124, 10172, 10344, 10345, 10357, 10451, 10460, 10647, 10681, 10703,
    10714, 10755, 10756, 10780, 10882, 10958, 11029, 11040, 11075, 11172,
    11205, 11413, 11439, 11528, 11547, 11673, 11682, 11715, 11730, 11875,
    11876, 11928, 11947, 11953, 12054, 12069, 12083, 12165, 12179, 12255,
    12261, 12295, 12321, 12533, 12677, 12704, 12715, 12816, 12830, 12867,
    12890, 12901, 12949, 13022, 13118, 13239, 13296, 13342, 13403, 13476,
    13619, 13640, 13641, 13659, 13666, 13696, 13781, 13803, 13822, 13877,
    13974, 14026, 14038, 14253, 14264, 14311, 14345, 14382, 14568, 14610,
    14725, 14757, 14821, 14997, 14999, 15042, 15125, 15128, 15212, 15257,
    15304, 15349, 15424, 15454, 15465, 15594, 15633, 15688, 15939, 15949,
    15961, 15989, 16151, 16214, 16261, 16332, 16348, 16508, 16513, 16588,
    16643, 16646, 16723, 16735, 16766, 16802, 16834, 16879, 16901, 16975,
    16998, 17018, 17030, 17054, 17057, 17075, 17119, 17166, 17224, 17275,
    17300, 17322, 17494, 17512, 17535, 17551, 17593, 17744, 17765, 17827,
    17881, 17883, 17954, 17959, 18018, 18062, 18153, 18183, 18184, 18236,
    18261, 18282, 18350, 18351, 18393, 18423, 18456, 18531, 18569, 18625,
    18674, 18699, 18718, 18815, 18941, 18983, 18996, 19018, 19039, 19059,
    19181, 19194, 19199, 19203, 19220, 19232, 19235, 19280, 19317, 19451,
    19468, 19478, 19508, 19690, 19705, 19725, 19737, 19771, 19858, 19923,
    19927, 19937, 19978, 20062, 20116, 20139, 20252, 20274, 20568, 20612,
    20658, 20691, 20709, 20798, 20826, 20979, 21045, 21134, 21245, 21281,
    21324, 21440, 21498, 21501, 21623, 21630, 21632, 21666, 21684, 21724,
    21793, 21801, 21813, 21897, 22018, 22206, 22240, 22292, 22314, 22381,
    22418, 22437, 22451, 22464, 22479, 22520, 22531, 22589, 22643, 22679,
    22728, 22773, 22788, 22952, 22996, 23122, 23137, 23151, 23172, 23223,
    23225, 23240, 23296, 23323, 23385, 23407, 23474, 23484, 23598, 23670,
    23695, 23774, 23803, 23806, 23861, 23872, 23934, 23975, 24113, 24235,
    24314, 24450, 24498, 24576, 24584, 24587, 24664, 24688, 24702, 24837,
    24869, 24883, 24969, 25030, 25046, 25075, 25077, 25111, 25153, 25190,
    25365, 25367, 25381, 25445, 25472, 25547, 25662, 25675, 25728, 25793,
    25812, 25813, 25831, 25841, 25872, 26003, 26012, 26029, 26072, 26089,
    26143, 26203, 26222, 26237, 26360, 26400, 26476, 26578, 26623, 26624,
    26641, 26668, 26754, 26756, 26870, 26890, 26905, 26940, 26953, 27052,
    27066, 27110, 27120, 27173, 27180, 27203, 27240, 27329, 27372, 27378,
    27398, 27422, 27551, 27567, 27582, 27716, 27747, 27761, 27763, 27785,
    27828, 27950, 28004, 28094, 28140, 28180, 28299, 28423, 28472, 28478,
    28482, 28486, 28520, 28587, 28634, 28644, 28714, 28730, 28745, 28749,
    28778, 28861, 28864, 28883, 28920, 28968, 28972, 28983, 29073, 29099,
    29131, 29157, 29191, 29222, 29242, 29415, 29535, 29541, 29547, 29615,
    29697, 29782, 29802, 30038, 30085, 30146, 30198, 30221, 30256, 30278,
    30303, 30432, 30455, 30515, 30537, 30561, 30566, 30592, 30661, 30665,
    30670, 30690, 30697, 30706, 30734, 30933, 30967, 31061, 31132, 31198,
    31227, 31232, 31309, 31391, 31400, 31445, 31486, 31637, 31715, 31751,
    31754, 31769, 31889, 31961, 32077, 32132, 32136, 32176, 32273, 32458,
    32468, 32482, 32798, 32814, 32834, 32872, 32910, 32937, 32940, 33026,
    33049, 33066, 33179, 33228, 33262, 33267, 33276, 33329, 33372, 33415,
    33508, 33582, 33605, 33619, 33688, 33772, 33781, 33792, 33865, 33866,
    33950, 33968, 34003, 34057, 34100, 34136, 34256, 34272, 34419, 34424,
    34455, 34611, 34613, 34617, 34641, 34651, 34754, 34829, 34900, 34905,
    34909, 35035, 35046, 35100, 35111, 35142, 35205, 35304, 35312, 35429,
    35464, 35514, 35576, 35678, 35799, 35841, 35862, 35896, 35904, 35974,
    36069, 36128, 36163, 36181, 36236, 36267, 36300, 36453, 36502, 36513,
    36613, 36710, 36754, 36822, 36975, 37101, 37121, 37138, 37154, 37243,
    37262, 37338, 37345, 37371, 37558, 37644, 37660, 37672, 37823, 37825,
    37861, 37919, 37935, 37988, 38089, 38119, 38120, 38161, 38235, 38281,
    38298, 38422, 38498, 38511, 38536, 38578, 38587, 38628, 38727, 38762,
    38786, 38839, 38978, 39017, 39039, 39057, 39104, 39134, 39147, 39331,
    39428, 39453, 39518, 39561, 39580, 39623, 39643, 39777, 39817, 39854,
    39862, 39889, 39958, 40104, 40150, 40157, 40160, 40222, 40275, 40315,
    40357, 40430, 40463, 40477, 40484, 40488, 40535, 40589, 40619, 40777,
    40781, 40814, 40835, 41059, 41092, 41152, 41165, 41309, 41349, 41586,
    41648, 41658, 41659, 41697, 41893, 41936, 41957, 41968, 42060, 42086,
    42153, 42187, 42190, 42231, 42346, 42360, 42509, 42547, 42553, 42578,
    42617, 42657, 42728, 42811, 42826, 42832, 42836, 42864, 42888, 42891,
    42909, 42994, 43099, 43144, 43166, 43226, 43231, 43236, 43253, 43377,
    43398, 43416, 43431, 43466, 43485, 43552, 43574, 43681, 43764, 43911,
    43943, 43980, 44160, 44215, 44251, 44274, 44279, 44326, 44366, 44414,
    44441, 44446, 44459, 44515, 44609, 44671, 44734, 44778, 44792, 44897,
    44960, 44980, 45090, 45178, 45194, 45261, 45306, 45337, 45353, 45493,
    45500, 45522, 45541, 45569, 45587, 45628, 45676, 45686, 45691, 45798,
    45887, 45893, 45920, 45930, 46058, 46113, 46146, 46147, 46230, 46257,
    46281, 46481, 46491, 46494, 46540, 46556, 46572, 46619, 46730, 46776,
    46833, 46850, 46899, 46983, 47049, 47059, 47062, 47121, 47260, 47311,
    47412, 47420, 47448, 47556, 47597, 47633, 47698, 47802, 47806, 47863,
    47870, 47912, 48014, 48021, 48038, 48114, 48116, 48150, 48247, 48335,
    48424, 48504, 48506, 48512, 48622, 48708, 48756, 48765, 48792, 48817,
    48865, 48894, 49077, 49081, 49161, 49180, 49210, 49256, 49266, 49315,
    49487, 49490, 49530, 49653, 49661, 49739, 49814, 49918, 49924, 49925, 49937
])


class CIFAR100HighInfData(CIFAR100Data):
    """
  CIFAR100Data wo High influence samples from vitaly
  or without a random number of datapoints which is the same size as the high influence points
  if train_highinf is False then it reverts to CIFAR100Data without 964 random samples.
  """

    def __init__(self, args):
        super().__init__(args)
        self.train_subset_mode = args.get('train_subset_mode',
                                          'high_influence')

    def _get_influence_indices(self):
        all_indices = np.arange(0, 50000)
        if self.train_subset_mode == 'high_influence':
            indices_to_remove = HIGH_INFLUENCE_CIFAR100
        elif self.train_subset_mode == 'random':
            indices_to_remove = np.random.choice(
                all_indices, size=len(HIGH_INFLUENCE_CIFAR100), replace=False)
        else:
            raise ValueError(
                f'Unknown train_subset_mode {self.train_subset_mode}')

        return np.setdiff1d(all_indices, indices_to_remove)

    def train_dataloader(self, shuffle=True, aug=True):
        """added optional shuffle parameter for generating random labels.
    added optional aug parameter to apply augmentation or not.

    """
        if (aug is True) and (self.train_aug is True):
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ])

        subset_indices = self._get_influence_indices()
        dataset = CIFAR100(
            root=self.hparams.data_dir,
            train=True,
            transform=transform,
            download=True,
        )
        # slice dataset
        dataset = Subset(dataset, subset_indices)

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader
