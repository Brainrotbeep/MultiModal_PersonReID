import random
from collections import defaultdict

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):

    def __init__(self, dataset, num_pids, num_instances):

        self.dataset = dataset
        self.num_pids = num_pids
        self.num_instances = num_instances

        self.index_dic = defaultdict(list)

        for index in range(len(dataset)):
            pid = dataset.samples[index]["pid"]
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        self.length = 0

        for pid in self.pids:

            idxs = self.index_dic[pid]

            num = len(idxs)

            if num < self.num_instances:
                num = self.num_instances

            self.length += num - num % self.num_instances

    def __iter__(self):

        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:

            idxs = self.index_dic[pid].copy()

            if len(idxs) < self.num_instances:

                idxs.extend(
                    random.choices(
                        idxs,
                        k=self.num_instances - len(idxs)
                    )
                )

            random.shuffle(idxs)

            batch = []

            for idx in idxs:

                batch.append(idx)

                if len(batch) == self.num_instances:

                    batch_idxs_dict[pid].append(batch)

                    batch = []

        avai_pids = self.pids.copy()

        final_idxs = []

        while len(avai_pids) >= self.num_pids:

            selected_pids = random.sample(
                avai_pids,
                self.num_pids
            )

            for pid in selected_pids:

                batch = batch_idxs_dict[pid].pop(0)

                final_idxs.extend(batch)

                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length