from networks.r2rnet import Region2RegionNet

# kwargs: Arguments for the given architecture
def build(arch, **kwargs):
    # Build protonet wrapper
    if arch == 'r2r_proto':
        net = Region2RegionNet(**kwargs)
    else:
        raise NotImplementedError('Unknown arch name ' + str(arch))

    return net
