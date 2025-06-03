from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_PEMS
from torch.utils.data import DataLoader
#from data_provider.uea import collate_fn
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        #if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
        #    batch_size = args.batch_size
        #else:
        #    batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    
    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers, #
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    elif args.task_name == 'short_term_forecast':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        data_set = Data(
        config=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        return data_set, data_loader

def collate_fn(batch, max_len):
    # Assuming batch is a list of tuples (input, target) and input is a sequence
    # Pad all sequences to the same length (max_len) with zeros
    inputs, targets = zip(*batch)
    
    # Pad sequences to max_len
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in inputs],
        batch_first=True, padding_value=0
    )[:, :max_len]  # Ensure padding to exactly max_len
    
    # Similarly, process targets (adjust as necessary)
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(target) for target in targets],
        batch_first=True, padding_value=0
    )[:, :max_len]
    
    return padded_inputs, padded_targets
