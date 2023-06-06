import itertools

import torch


def _sim_mat(t1, t2, temperature, device, batch_size, eps=1e-8):
    if t1 is None or t2 is None:
        return torch.zeros(batch_size, batch_size, device=device)

    norm_1 = t1.norm(dim=1)[:, None]
    norm_2 = t2.norm(dim=1)[:, None]

    norm_t1 = t1 / torch.max(norm_1, eps * torch.ones_like(norm_1))
    norm_t2 = t2 / torch.max(norm_2, eps * torch.ones_like(norm_2))
    similarity_matrix = torch.mm(norm_t1, norm_t2.transpose(0, 1))

    assert similarity_matrix.shape == (t1.shape[0], t2.shape[0])

    matrix = torch.exp(similarity_matrix / temperature)
    return matrix


def _latent_positive_sampling(embeddings, device, seed, sampling_factor=1, sampling_method: str = 'random'):
    batch_size = embeddings.shape[0]

    augmentation_list = []
    augmentation_record_dict = dict()
    if sampling_method == 'random':
        raise NotImplementedError

    elif sampling_method == 'complete':
        for i in range(batch_size):
            for j in range(batch_size):
                augmentation = torch.mean(torch.stack((embeddings[i], embeddings[j])), 0, True)
                augmentation_list.append(augmentation)
                # collect augmentation derived from batch element. Will be used for element anomaly score calculation
                augmentation_record_dict[i] = augmentation_record_dict[i] + [
                    len(augmentation_list) - 1] if i in augmentation_record_dict.keys() else [
                    len(augmentation_list) - 1]
                augmentation_record_dict[j] = augmentation_record_dict[j] + [
                    len(augmentation_list) - 1] if j in augmentation_record_dict.keys() else [
                    len(augmentation_list) - 1]

        latent_positive_augmentation = torch.squeeze(torch.stack(augmentation_list)).to(device)

    else:
        raise NotImplementedError

    return latent_positive_augmentation, augmentation_record_dict


def _elementwise_average(matrix, batch_size):
    buf = []
    sim_vec = torch.sum(matrix, axis=1)

    for i in range(batch_size):
        buf.append(sim_vec[i:][::batch_size].mean(axis=0))

    return torch.tensor(buf, device=matrix.device)


def combinations(i, j):
    buf = []
    for a, b in itertools.product([x for x in range(i)], [y for y in range(j)]):
        if (a, b) and (b, a) not in buf:
            buf.append((a, b))
    return buf

def contrastive_loss_func_new(X_embeddings, negative_augmentation_embedding_list, positive_augmentation_embedding_list,
                          temperature, seed, latent_augmentation=False, type='training_loss'):

    batch_size = X_embeddings.shape[0]
    device = X_embeddings.device
    # prepare similarity matrix
    x_x_sim_mat = _sim_mat(X_embeddings, X_embeddings, temperature, device, batch_size)
    if len(negative_augmentation_embedding_list) != 0:
        n_n_sim_mats = [_sim_mat(negative_augmentation_embeddings, negative_augmentation_embeddings, temperature, device,
                                 batch_size) for negative_augmentation_embeddings in negative_augmentation_embedding_list]
        x_n_sim_mats = [_sim_mat(X_embeddings, negative_augmentation_embeddings, temperature, device, batch_size) for
                        negative_augmentation_embeddings in negative_augmentation_embedding_list]
    if len(positive_augmentation_embedding_list) != 0:
        p_p_sim_mats = [_sim_mat(positive_augmentation_embeddings, positive_augmentation_embeddings, temperature, device,
                                 batch_size) for positive_augmentation_embeddings in positive_augmentation_embedding_list]
        x_p_sim_mats = [_sim_mat(X_embeddings, positive_augmentation_embeddings, temperature, device, batch_size) for
                        positive_augmentation_embeddings in positive_augmentation_embedding_list]

    if len(negative_augmentation_embedding_list) != 0 and len(positive_augmentation_embedding_list) != 0:
        n_p_sim_mats = [
            _sim_mat(negative_augmentation_embedding_list[i], positive_augmentation_embedding_list[j], temperature, device,
                     batch_size) for (i, j) in combinations(len(negative_augmentation_embedding_list),
                                                            len(positive_augmentation_embedding_list))]

    if latent_augmentation:
        latent_positive_augmentation, augmentation_record_dict = _latent_positive_sampling(X_embeddings, device, seed,
                                                                                           sampling_method='complete')
        l_l_sim_mat = _sim_mat(latent_positive_augmentation, latent_positive_augmentation, temperature, device,
                               batch_size)
        x_l_sim_mat = _sim_mat(X_embeddings, latent_positive_augmentation, temperature, device, batch_size)

    # calculate training loss
    if type == 'training_loss':
        negative_component = x_x_sim_mat.mul(torch.eye(x_x_sim_mat.shape[0], device=x_x_sim_mat.device)).mean()

        if len(positive_augmentation_embedding_list) != 0:
            positive_component = torch.stack([x_p_sim_mat.diag().mean() for x_p_sim_mat in x_p_sim_mats]).mean()
            negative_component += torch.stack([x_p_sim_mat.mul(torch.eye(x_p_sim_mat.shape[0], device=x_p_sim_mat.device)) for x_p_sim_mat in x_p_sim_mats]).mean()
            negative_component += torch.stack([p_p_sim_mat.diag().mean() for p_p_sim_mat in p_p_sim_mats]).mean()
        if len(negative_augmentation_embedding_list) != 0:
            negative_component += torch.stack([x_n_sim_mat.mul(torch.eye(x_n_sim_mat.shape[0], device=x_n_sim_mat.device)) for x_n_sim_mat in x_n_sim_mats]).mean()
            negative_component += torch.stack([n_n_sim_mat.mul(torch.eye(n_n_sim_mat.shape[0], device=n_n_sim_mat.device)) for n_n_sim_mat in n_n_sim_mats]).mean()


    elif type == 'anomaly_score':

        negative_component = x_x_sim_mat.mul(torch.eye(x_x_sim_mat.shape[0], device=x_x_sim_mat.device)).mean(axis=0)

        if len(positive_augmentation_embedding_list) != 0:
            positive_component = torch.stack([x_p_sim_mat.diag().mean() for x_p_sim_mat in x_p_sim_mats]).mean(axis=0)
            negative_component += torch.stack([x_p_sim_mat.mul(torch.eye(x_p_sim_mat.shape[0], device=x_p_sim_mat.device)).mean(axis=1) for x_p_sim_mat in x_p_sim_mats]).mean(axis=0)
            negative_component += torch.stack([p_p_sim_mat.diag().mean() for p_p_sim_mat in p_p_sim_mats]).mean(axis=0)
        if len(negative_augmentation_embedding_list) != 0:
            negative_component += torch.stack([x_n_sim_mat.mul(torch.eye(x_n_sim_mat.shape[0], device=x_n_sim_mat.device)).mean(axis=1) for x_n_sim_mat in x_n_sim_mats]).mean(axis=0)
            negative_component += torch.stack([n_n_sim_mat.mul(torch.eye(n_n_sim_mat.shape[0], device=n_n_sim_mat.device)).mean(axis=1) for n_n_sim_mat in n_n_sim_mats]).mean(axis=0)
    else:
        raise NotImplementedError
    loss = -torch.log(positive_component / (
            positive_component + negative_component)) if len(negative_augmentation_embedding_list) != 0 \
        else -torch.log(positive_component)

    return loss


def contrastive_loss_func(X_embeddings, negative_augmentation_embedding_list, positive_augmentation_embedding_list,
                          temperature, seed, latent_augmentation=False, type='training_loss'):
    """

    :param augmented_embeddings: (batch_size, hidden_dim)
    :param temperature:
    :param latent_augmentation:
    :return:
    """

    batch_size = X_embeddings.shape[0]
    device = X_embeddings.device
    # prepare similarity matrix
    x_x_sim_mat = _sim_mat(X_embeddings, X_embeddings, temperature, device, batch_size)
    if len(negative_augmentation_embedding_list) != 0:
        n_n_sim_mats = [_sim_mat(negative_augmentation_embeddings, negative_augmentation_embeddings, temperature, device,
                                 batch_size) for negative_augmentation_embeddings in negative_augmentation_embedding_list]
        x_n_sim_mats = [_sim_mat(X_embeddings, negative_augmentation_embeddings, temperature, device, batch_size) for
                        negative_augmentation_embeddings in negative_augmentation_embedding_list]
    if len(positive_augmentation_embedding_list) != 0:
        p_p_sim_mats = [_sim_mat(positive_augmentation_embeddings, positive_augmentation_embeddings, temperature, device,
                                 batch_size) for positive_augmentation_embeddings in positive_augmentation_embedding_list]
        x_p_sim_mats = [_sim_mat(X_embeddings, positive_augmentation_embeddings, temperature, device, batch_size) for
                        positive_augmentation_embeddings in positive_augmentation_embedding_list]

    if len(negative_augmentation_embedding_list) != 0 and len(positive_augmentation_embedding_list) != 0:
        n_p_sim_mats = [
            _sim_mat(negative_augmentation_embedding_list[i], positive_augmentation_embedding_list[j], temperature, device,
                     batch_size) for (i, j) in combinations(len(negative_augmentation_embedding_list),
                                                            len(positive_augmentation_embedding_list))]

    if latent_augmentation:
        latent_positive_augmentation, augmentation_record_dict = _latent_positive_sampling(X_embeddings, device, seed,
                                                                                           sampling_method='complete')
        l_l_sim_mat = _sim_mat(latent_positive_augmentation, latent_positive_augmentation, temperature, device,
                               batch_size)
        x_l_sim_mat = _sim_mat(X_embeddings, latent_positive_augmentation, temperature, device, batch_size)

    # calculate training loss
    if type == 'training_loss':
        if len(positive_augmentation_embedding_list) != 0:
            positive_component = torch.stack([x_p_sim_mat.diag().mean() for x_p_sim_mat in x_p_sim_mats]).mean()
            negative_component = torch.stack(
                [x_p_sim_mat.mul(torch.eye(x_p_sim_mat.shape[0], device=x_p_sim_mat.device)).mean() for x_p_sim_mat in x_p_sim_mats]).mean()

            if len(negative_augmentation_embedding_list) != 0:
                negative_component += torch.stack([x_n_sim_mat.mean() for x_n_sim_mat in x_n_sim_mats]).mean()

    # calculate anomaly score
    elif type == 'anomaly_score':
        if len(positive_augmentation_embedding_list) != 0:
            positive_component = torch.stack([x_p_sim_mat.diag() for x_p_sim_mat in x_p_sim_mats]).mean(axis=0)
            negative_component = torch.stack(
                [x_p_sim_mat.mul(torch.eye(x_p_sim_mat.shape[0], device=x_p_sim_mat.device)).mean(axis=1) for x_p_sim_mat in x_p_sim_mats]).mean(axis=0)
            if len(negative_augmentation_embedding_list) != 0:
                negative_component += torch.stack([x_n_sim_mat.mean(axis=1) for x_n_sim_mat in x_n_sim_mats]).mean(axis=0)
    else:
        raise NotImplementedError
    loss = -torch.log(positive_component / (
            positive_component + negative_component)) if len(negative_augmentation_embedding_list) != 0 \
        else -torch.log(positive_component)

    return loss


def contrastive_loss_func_old(X_embeddings, negative_augmentation_embedding_list, positive_augmentation_embedding_list,
                          temperature, seed, latent_augmentation=False, type='training_loss'):
    """

    :param augmented_embeddings: (batch_size, hidden_dim)
    :param temperature:
    :param latent_augmentation:
    :return:
    """

    batch_size = X_embeddings.shape[0]
    device = X_embeddings.device
    # prepare similarity matrix
    x_x_sim_mat = _sim_mat(X_embeddings, X_embeddings, temperature, device, batch_size)
    if len(negative_augmentation_embedding_list) != 0:
        n_n_sim_mats = [_sim_mat(negative_augmentation_embeddings, negative_augmentation_embeddings, temperature, device,
                                 batch_size) for negative_augmentation_embeddings in negative_augmentation_embedding_list]
        x_n_sim_mats = [_sim_mat(X_embeddings, negative_augmentation_embeddings, temperature, device, batch_size) for
                        negative_augmentation_embeddings in negative_augmentation_embedding_list]
    if len(positive_augmentation_embedding_list) != 0:
        p_p_sim_mats = [_sim_mat(positive_augmentation_embeddings, positive_augmentation_embeddings, temperature, device,
                                 batch_size) for positive_augmentation_embeddings in positive_augmentation_embedding_list]
        x_p_sim_mats = [_sim_mat(X_embeddings, positive_augmentation_embeddings, temperature, device, batch_size) for
                        positive_augmentation_embeddings in positive_augmentation_embedding_list]

    if len(negative_augmentation_embedding_list) != 0 and len(positive_augmentation_embedding_list) != 0:
        n_p_sim_mats = [
            _sim_mat(negative_augmentation_embedding_list[i], positive_augmentation_embedding_list[j], temperature, device,
                     batch_size) for (i, j) in combinations(len(negative_augmentation_embedding_list),
                                                            len(positive_augmentation_embedding_list))]

    if latent_augmentation:
        latent_positive_augmentation, augmentation_record_dict = _latent_positive_sampling(X_embeddings, device, seed,
                                                                                           sampling_method='complete')
        l_l_sim_mat = _sim_mat(latent_positive_augmentation, latent_positive_augmentation, temperature, device,
                               batch_size)
        x_l_sim_mat = _sim_mat(X_embeddings, latent_positive_augmentation, temperature, device, batch_size)

    # calculate training loss
    if type == 'training_loss':
        positive_component = x_x_sim_mat.mean()
        negative_component = torch.tensor(0., device=positive_component.device)

        if len(positive_augmentation_embedding_list) != 0:
            positive_component += (torch.stack([x_p_sim_mat.mean() for x_p_sim_mat in x_p_sim_mats]).mean() \
                             + torch.stack([p_p_sim_mat.mean() for p_p_sim_mat in p_p_sim_mats]).mean())
        if latent_augmentation:
            positive_component += l_l_sim_mat.mean() + x_l_sim_mat.mean()
        if len(negative_augmentation_embedding_list) != 0:
            negative_component += torch.stack([n_n_sim_mat.mean() for n_n_sim_mat in n_n_sim_mats]).mean()
            negative_component += torch.stack([x_n_sim_mat.mean() for x_n_sim_mat in x_n_sim_mats]).mean()
            if len(positive_augmentation_embedding_list) != 0:
                negative_component += torch.stack([n_p_sim_mat.mean() for n_p_sim_mat in n_p_sim_mats]).mean()
    # calculate anomaly score
    elif type == 'anomaly_score':
        positive_component = torch.mean(x_x_sim_mat, axis=1)
        negative_component = torch.zeros(positive_component.shape, device=positive_component.device)
        if len(positive_augmentation_embedding_list) != 0:
            positive_component += (torch.stack([torch.mean(x_p_sim_mat, axis=1) for x_p_sim_mat in x_p_sim_mats]).mean(axis=0) + \
                                    torch.stack([torch.mean(p_p_sim_mat, axis=1) for p_p_sim_mat in p_p_sim_mats]).mean(axis=0))
        if latent_augmentation:  # todo: can be optimized
            score_buf = []
            latent_sim_vec = torch.mean(l_l_sim_mat, axis=1)

            for i in range(batch_size):
                instance_relevant_augmentations_idx = augmentation_record_dict[i]
                score_buf.append(torch.mean(latent_sim_vec[instance_relevant_augmentations_idx]))
            batch_scores = torch.stack(score_buf).reshape(-1, )
            positive_component += batch_scores

            positive_component += torch.mean(x_l_sim_mat, axis=1)
        if len(negative_augmentation_embedding_list) != 0:
            negative_component += torch.stack([torch.mean(n_n_sim_mat, axis=1) for n_n_sim_mat in n_n_sim_mats]).mean(axis=0)
            negative_component += torch.stack([torch.mean(x_n_sim_mat, axis=1) for x_n_sim_mat in x_n_sim_mats]).mean(axis=0)
            if len(positive_augmentation_embedding_list) != 0:
                negative_component += torch.stack([torch.mean(n_p_sim_mat, axis=1) for n_p_sim_mat in n_p_sim_mats]).mean(axis=0)
    else:
        raise NotImplementedError

    loss = -torch.log(positive_component / (
            positive_component + negative_component)) if len(negative_augmentation_embedding_list) != 0 \
        else -torch.log(positive_component)

    return loss
