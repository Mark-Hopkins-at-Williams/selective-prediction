import torch


def confuse_two(labels):
    labels = labels.clone()
    one_and_sevens = (labels == 1) + (labels == 7)
    one_seven_shape = labels[one_and_sevens].shape
    # change the second argument for different weights
    new_labels = torch.randint(0, 2, one_seven_shape) 
    new_labels[new_labels == 0] = 7    
    labels[one_and_sevens] = new_labels
    return labels


def confuse_all(labels):
    labels = confuse_two(labels)
    one_and_sevens = (labels == 2) + (labels == 3)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 2, one_seven_shape)
    new_labels[new_labels > 0] = 3
    new_labels[new_labels == 0] = 2
    labels[one_and_sevens] = new_labels
    one_and_sevens = (labels == 4) + (labels == 5)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 4, one_seven_shape)
    new_labels[new_labels > 0] = 4
    new_labels[new_labels == 0] = 5
    labels[one_and_sevens] = new_labels
    return labels        


confuser_lookup = {'two': confuse_two,
                   'all': confuse_all}


class MnistLoader:
    
    def __init__(self, dataset, bsz=64, shuffle = True, confuser=lambda x: x):
        self.loader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=bsz, 
                                                  shuffle=shuffle)
        self.confuser = confuser
        
    def __iter__(self):
        for images, labels in self.loader:
            images = images.view(images.shape[0], -1)
            labels = self.confuser(labels)
            yield images, labels

    def __len__(self):
        return len(self.loader)


class ConfusedMnistLoader(MnistLoader):
    
    def __init__(self, dataset, bsz=64, confuser='all', shuffle=True):
        super().__init__(dataset, bsz, shuffle, confuser_lookup[confuser])
        

class MnistPairLoader:
    def __init__(self, dataset, bsz=64, shuffle=True, confuser=lambda x:x):
        self.bsz = bsz
        self.dataset = dataset
        self.single_img_loader1 = torch.utils.data.DataLoader(dataset, 
                                                              batch_size=bsz, 
                                                              shuffle=shuffle)
        self.single_img_loader2 = torch.utils.data.DataLoader(dataset, 
                                                              batch_size=bsz, 
                                                              shuffle=shuffle)
        assert(len(self.single_img_loader1) == len(self.single_img_loader2))
        self.confuser = confuser
    
    def __len__(self):
        return len(self.single_img_loader1)

    def __iter__(self):
        for ((imgs1, lbls1), (imgs2, lbls2)) in zip(self.single_img_loader1, 
                                                    self.single_img_loader2):
            lbls1 = self.confuser(lbls1)
            lbls2 = self.confuser(lbls2)
            imgs1 = imgs1.view(imgs1.shape[0], -1)
            imgs2 = imgs2.view(imgs2.shape[0], -1)
            yield imgs1, imgs2, lbls1, lbls2


class ConfusedMnistPairLoader(MnistPairLoader):
    def __init__(self, dataset, bsz=64, confuser='all', shuffle=True):
        super().__init__(dataset, bsz, shuffle, confuser_lookup[confuser])
