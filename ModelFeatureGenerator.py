import deepchem as dc


def generate_model_feature(m_name, n_tasks, args):
    op_dir = f"{args['model_dir']}/{args['split']}/{args['mode']}"
    mode = args['mode']
    if mode == 'soft':
        mode = 'classification'
    batch_size = args['batch_size']
    if m_name == 'GCN':
        feat = dc.feat.MolGraphConvFeaturizer()
        net = dc.models.GCNModel(n_tasks=n_tasks, mode=mode, model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
    elif m_name == 'GAT':
        feat = dc.feat.MolGraphConvFeaturizer()
        net = dc.models.GATModel(n_tasks=n_tasks, mode=mode, model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
    elif m_name == 'AttentiveFP':
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        net = dc.models.AttentiveFPModel(n_tasks=n_tasks, mode=mode,
                                         model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
    elif m_name == 'MPNN':
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        net = dc.models.torch_models.MPNNModel(n_tasks=n_tasks, mode=mode,
                                               model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
    elif m_name == 'PAGTN':
        feat = dc.feat.PagtnMolGraphFeaturizer()
        net = dc.models.PagtnModel(n_tasks=n_tasks, mode=mode, model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
    # TODO add 'n_classes-2' in DMPNN, may need to remove when using regression mode
    elif m_name == 'DMPNN':
        feat = dc.feat.DMPNNFeaturizer()
        if mode == 'classification':
            net = dc.models.DMPNNModel(n_tasks=n_tasks, n_classes=2, mode=mode,
                                       model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
        else:
            net = dc.models.DMPNNModel(n_tasks=n_tasks, mode=mode, model_dir=f"{op_dir}/{m_name}", batch_size=batch_size)
    else:
        quit(f'Unknown model name: {m_name}')
    return feat, net
