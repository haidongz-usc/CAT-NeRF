import torch

def pearson(latent, net):
    eps = 1e-5
    lat_len = torch.range(0, latent.num_embeddings - 1).long().to(torch.device('cuda'))
    latent_val = latent(lat_len)
    avg_lat = torch.mean(latent_val, 0)
    diff_lat = latent_val - avg_lat
    cov = torch.prod(torch.pow(torch.abs(diff_lat), 1/64), 1).sum()
    sq_diff_lat = torch.std(diff_lat, 0)
    gamma = torch.prod(torch.pow(sq_diff_lat, 1/64)) + eps
    return cov/gamma *1e-3#* 1e13


def cov(latent, net):
    lat_len = torch.range(0, latent.num_embeddings - 1).long().to(torch.device('cuda'))
    latent_val = latent(lat_len)
    cov_val = torch.abs(torch.cov(latent_val))
    return (torch.sum(cov_val) - torch.sum(torch.diag(cov_val, 0))) / (latent.num_embeddings - 1) / (latent.num_embeddings - 1)

def cov_pose(latent1, latent2, net):
    lat_len = torch.range(0, latent1.num_embeddings - 1).long().to(torch.device('cuda'))
    latent1_val = latent1(lat_len)
    lat_len = torch.range(0, latent2.num_embeddings - 1).long().to(torch.device('cuda'))
    latent2_val = latent2(lat_len)
    latent_val = torch.cat([latent1_val, latent2_val], 0)
    cov_val = torch.abs(torch.cov(latent_val))
    latdim = latent1.num_embeddings + latent2.num_embeddings - 1

    return (torch.sum(cov_val) - torch.sum(torch.diag(cov_val, 0))) / latdim / latdim

def global_sim(latent1, latent2):
    lat_id = torch.Tensor([0]).long().to(torch.device('cuda'))
    latent1_val = latent1(lat_id)
    latent2_val = latent2(lat_id)
    return torch.nn.functional.smooth_l1_loss(latent1_val, latent2_val)

