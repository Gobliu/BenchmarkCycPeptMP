import deepchem as dc


def generate_model_feature(m_name, op_dir, batch_size):
    if m_name == 'GCN':
        feat = dc.feat.MolGraphConvFeaturizer()
        net = dc.models.GCNModel(n_tasks=1, mode='regression', model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    elif m_name == 'GAT':
        feat = dc.feat.MolGraphConvFeaturizer()
        net = dc.models.GATModel(n_tasks=1, mode='regression', model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    elif m_name == 'AttentiveFP':
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        net = dc.models.AttentiveFPModel(n_tasks=1, mode='regression',
                                         model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    elif m_name == 'MPNN':
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        net = dc.models.torch_models.MPNNModel(n_tasks=1, mode='regression',
                                               model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    elif m_name == 'PAGTN':
        feat = dc.feat.PagtnMolGraphFeaturizer()
        net = dc.models.PagtnModel(n_tasks=1, mode='regression', model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    elif m_name == 'DMPNN':
        feat = dc.feat.DMPNNFeaturizer()
        net = dc.models.DMPNNModel(n_tasks=1, mode='regression', model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    # elif m_name == 'AtomicConvModel':
    #     featurizer = dc.feat.AtomicConvFeaturizer()
    #     net = dc.models.AtomicConvModel(n_tasks=1, mode='regression',
    #                                     model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    # elif m_name == 'ChemCeption':
    #     featurizer = dc.feat.SmilesToImage()
    #     net = dc.models.ChemCeption(n_tasks=1, mode='regression',
    #                                 model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    # elif m_name == 'DAGModel':
    #     featurizer = dc.feat.ConvMolFeaturizer()
    #     net = dc.models.DAGModel(n_tasks=1, mode='regression', model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    # elif m_name == 'GraphConvModel':
    #     featurizer = dc.feat.ConvMolFeaturizer()
    #     net = dc.models.GraphConvModel(n_tasks=1, mode='regression',
    #                                    model_dir=f"{op_dir}{m_name}", batch_size=batch_size)
    else:
        quit(f'Unknown model name: {m_name}')
    return feat, net
