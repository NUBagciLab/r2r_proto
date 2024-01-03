from reader import cxr14_reader

def get_class_name(opt):
    dataset = opt['dataset']

    if dataset['name'] == 'cxr-14':
        if dataset['add_healty_as_label']:
            str2labelid = cxr14_reader.LABEL_MAPPING_WITH_HEALTY
            labelid2str = {v:k for k, v in str2labelid.items()}
        else:
            str2labelid = cxr14_reader.LABEL_MAPPING
            labelid2str = {v:k for k, v in str2labelid.items()}

    else:
        raise NotImplementedError('Unknown dataset name ' + str(dataset['name']))
    

    return str2labelid, labelid2str
            