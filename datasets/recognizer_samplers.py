from pytorch_metric_learning.utils import common_functions as c_f
from torch.utils.data.sampler import Sampler
import torch


class FocusedNoReplacementMPerClassSampler(Sampler):

    def __init__(self, dataset, m, batch_size):
        labels = dataset.targets
        assert not batch_size is None, "Batch size is None!"
        if isinstance(labels, torch.Tensor): labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.dataset_len = self.m_per_class * len(self.labels)
        assert self.dataset_len >= self.batch_size
        assert self.length_of_single_pass >= self.batch_size, f"m * (number of unique labels ({len(self.labels)}) must be >= batch_size"
        assert self.batch_size % self.m_per_class == 0, "m_per_class must divide batch_size without any remainder"
        self.dataset_len -= self.dataset_len % self.batch_size

    def __len__(self):
        return self.dataset_len

    def __iter__(self):

        idx_list = [0] * self.dataset_len
        i = 0
        c_f.NUMPY_RANDOM.shuffle(self.labels)

        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(self.labels_to_indices[label])

        for label in self.labels:
            t = list(indices_remaining_dict[label])
            if len(t) == 0:
                randchoice = c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class).tolist()
            elif len(t) < self.m_per_class:
                randchoice = t + c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
            else:
                randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
            indices_remaining_dict[label] -= set(randchoice)
            idx_list[i : i + self.m_per_class] = randchoice
            i += self.m_per_class
        
        assert i == self.dataset_len, f"i is {i}, dataset len is {self.dataset_len}"
        
        notseen_count = 0
        for k in indices_remaining_dict.keys():
            notseen_count += len(indices_remaining_dict[k])
        print(f"Samples not seen: {notseen_count}")

        return iter(idx_list)


class ExhaustiveMPerClassSampler(Sampler):

    def __init__(self, dataset, m, batch_size):
        assert not batch_size is None, "Batch size is None!"
        self.labels = dataset.targets
        if isinstance(self.labels, torch.Tensor): self.labels = self.labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.labels_to_indices = c_f.get_labels_to_indices(self.labels)
        self.labels = list(self.labels_to_indices.keys())
        assert self.batch_size % self.m_per_class == 0, "m_per_class must divide batch_size without any remainder"

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):

        idx_list = []
        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(self.labels_to_indices[label])

        while sum(len(v) for k, v in indices_remaining_dict.items()) != 0:
            for label in self.labels:
                t = list(indices_remaining_dict[label])
                if len(t) == 0:
                    continue
                if len(t) < self.m_per_class:
                    randchoice = t + c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
                else:
                    randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
                assert len(randchoice) == self.m_per_class
                indices_remaining_dict[label] -= set(randchoice)
                idx_list.extend(randchoice)

        list_of_lists = [idx_list[i:i + self.m_per_class] for i in range(0, len(idx_list), self.m_per_class)]
        c_f.NUMPY_RANDOM.shuffle(list_of_lists)
        idx_list = sum(list_of_lists, [])

        print(f"Number of samples in epoch: {len(idx_list)}")

        return iter(idx_list)


class NoReplacementMPerClassSampler(Sampler):

    def __init__(self, dataset, m, batch_size, num_passes):
        labels = dataset.targets
        assert not batch_size is None, "Batch size is None!"
        if isinstance(labels, torch.Tensor): labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.dataset_len = int(self.length_of_single_pass * num_passes) # int(math.ceil(len(dataset) / batch_size)) * batch_size
        assert self.dataset_len >= self.batch_size
        assert self.length_of_single_pass >= self.batch_size, f"m * (number of unique labels ({len(self.labels)}) must be >= batch_size"
        assert self.batch_size % self.m_per_class == 0, "m_per_class must divide batch_size without any remainder"
        self.dataset_len -= self.dataset_len % self.batch_size

    def __len__(self):
        return self.dataset_len

    def __iter__(self):

        idx_list = [0] * self.dataset_len
        i = 0; j = 0
        num_batches = self.calculate_num_batches()
        num_classes_per_batch = self.batch_size // self.m_per_class
        c_f.NUMPY_RANDOM.shuffle(self.labels)

        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(self.labels_to_indices[label])

        for _ in range(num_batches):
            curr_label_set = self.labels[j : j + num_classes_per_batch]
            j += num_classes_per_batch
            assert len(curr_label_set) == num_classes_per_batch, f"{j}, {len(self.labels)}"
            if j + num_classes_per_batch >= len(self.labels):
                print(f"All unique labels/classes batched, {len(self.labels)}; restarting...")
                c_f.NUMPY_RANDOM.shuffle(self.labels)
                j = 0
            for label in curr_label_set:
                t = list(indices_remaining_dict[label])
                if len(t) == 0:
                    randchoice = c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class).tolist()
                elif len(t) < self.m_per_class:
                    randchoice = t + c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
                else:
                    randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
                indices_remaining_dict[label] -= set(randchoice)
                idx_list[i : i + self.m_per_class] = randchoice
                i += self.m_per_class
        
        notseen_count = 0
        for k in indices_remaining_dict.keys():
            notseen_count += len(indices_remaining_dict[k])
        print(f"Samples not seen: {notseen_count}")

        return iter(idx_list)

    def calculate_num_batches(self):
        assert self.batch_size < self.dataset_len, "Batch size is larger than dataset!"
        return self.dataset_len // self.batch_size


class HardNegativeClassSampler(Sampler):

    def __init__(self, 
            dataset, 
            classidx, 
            hardnegs, 
            hnset_per_batch=1, 
            m=4, 
            batch_size=128, 
            hns_set_size=8,
            num_passes=1
        ):

        labels = dataset.targets

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        self.labels = labels
        print(f"Number of hard negative sets: {len(hardnegs)}")

        self.all_labels_for_negs = []
        for hns in hardnegs:
            lab_neg_set = [classidx[str(ord(c))] for c in hns]
            assert len(lab_neg_set) == hns_set_size
            self.all_labels_for_negs.append(lab_neg_set)
        
        self.batch_size = batch_size
        self.m_per_class = m
        self.hnset_per_batch = hnset_per_batch

        self._sampler = NoReplacementMPerClassSampler(
            dataset=dataset, m=m, batch_size=batch_size, num_passes=num_passes
        )

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):

        _idx_list = list(self._sampler.__iter__())
        c_f.NUMPY_RANDOM.shuffle(self.all_labels_for_negs)
        labels_to_indices = c_f.get_labels_to_indices(self.labels)
        all_hn_indices = []

        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(labels_to_indices[label])

        for hn_labels_for_batch in self.all_labels_for_negs:
            hn_idx_for_batch = []
            for label in hn_labels_for_batch:
                t = list(indices_remaining_dict[label])
                if len(t) == 0:
                    t = labels_to_indices[label].tolist()
                if len(t) != 0: # label/underlying char is in eval set...
                    if len(t) < self.m_per_class:
                        randchoice = t + c_f.safe_random_choice(labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
                    else:
                        randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
                    assert len(randchoice) == self.m_per_class
                    indices_remaining_dict[label] -= set(randchoice)
                    hn_idx_for_batch.extend(randchoice)
            all_hn_indices.append(hn_idx_for_batch)

        """
        for bidx in range(0, len(_idx_list), self.batch_size):
            hnidx = c_f.NUMPY_RANDOM.choice(range(len(all_hn_indices)))
            _idx_list[bidx:bidx] = all_hn_indices[hnidx]
        """

        """
        list_of_lists = [_idx_list[i:i + self.m_per_class] for i in range(0, len(_idx_list), self.m_per_class)]
        c_f.NUMPY_RANDOM.shuffle(list_of_lists)
        _idx_list = sum(list_of_lists, [])
        """
        
        for hni in all_hn_indices:
            ridx = c_f.NUMPY_RANDOM.choice(range(0, len(_idx_list), self.batch_size))
            _idx_list[ridx:ridx] = hni

        print(f"Number of samples in epoch (hard negatives): {len(_idx_list)}")
        
        return iter(_idx_list)
