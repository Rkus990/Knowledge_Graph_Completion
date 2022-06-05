from __future__ import division
import src.param as param
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class KGFunction(nn.Module):
    def __init__(self, device,num_entity,num_relation):
        super(KGFunction, self).__init__()
        # KG model
        self.device = device
        self.num_entity = num_entity
        self.num_relation = num_relation

        # Entity and Relation Initialization
        if param.knowledge == 'rotate':  # double embedding
            emd_range = param.rotate_embedding_range()

            self.entity_embedding_layer = nn.Parameter(torch.zeros(num_entity, param.dim))
            nn.init.uniform_(
                tensor=self.entity_embedding_layer,
                a=-emd_range,
                b=emd_range
            )

            self.rel_embedding_layer = nn.Parameter(torch.zeros(num_relation, int(param.dim / 2)))
            nn.init.uniform_(
                tensor=self.rel_embedding_layer,
                a=-emd_range,
                b=emd_range
            )

        else:
            emd_range = param.rotate_embedding_range()
            # self.entity_embedding_layer = nn.Parameter(torch.zeros(num_entity, param.dim))
            self.entity_embedding_layer = nn.Embedding(num_entity, param.dim)
            # nn.init.uniform_(
            #     tensor=self.entity_embedding_layer,
            #     a=-emd_range,
            #     b=emd_range
            # )

            nn.init.xavier_normal_(self.entity_embedding_layer.weight)


            # self.rel_embedding_layer = nn.Parameter(torch.zeros(num_relation, param.dim))
            self.rel_embedding_layer = nn.Embedding(num_relation, param.dim)
            # nn.init.uniform_(
            #     tensor=self.rel_embedding_layer,
            #     a=-emd_range,
            #     b=emd_range
            # )

            nn.init.xavier_normal_(self.rel_embedding_layer.weight)
            self.criterion = nn.MarginRankingLoss(margin=param.margin, reduction='mean')


    def get_negative_samples(self,batch_size_each):
        neg_all = []
        for _ in range(param.neg_per_pos):
            rand_negs = torch.randint(high=self.num_entity, size=(batch_size_each,),
                                  device=self.device)  # [b,num_neg]

            neg_each = self.entity_embedding_layer(rand_negs).unsqueeze(1)

            neg_all.append(neg_each)

        neg_all = torch.cat(neg_all,dim = 1)


        return neg_all


    def forward(self, sample):
        batch_size_each = sample.size()[0]

        # h = torch.index_select(
        #     self.entity_embedding_layer,
        #     dim=0,
        #     index=sample[:, 0]
        # ).unsqueeze(1)

        h = self.entity_embedding_layer(sample[:,0]).unsqueeze(1)

        # r = torch.index_select(
        #     self.rel_embedding_layer,
        #     dim=0,
        #     index=sample[:, 1]
        # ).unsqueeze(1)

        r = self.rel_embedding_layer(sample[:, 1]).unsqueeze(1)

        # t = torch.index_select(
        #     self.entity_embedding_layer,
        #     dim=0,
        #     index=sample[:, 2]
        # ).unsqueeze(1)
        t = self.entity_embedding_layer(sample[:, 2]).unsqueeze(1)


        projected_t = project_t([h, r],self.device) ##
        pos_loss = define_loss([t, projected_t]) ## [b,1]

        # negative sampling TODO: has problem. Not real negative samples, and batch elements share the same negative sampling.
        # TODO: change into each element has a unique negative sampling & real negative sapmles.
        # TODO: generate negative sample once and keep it fixed


        neg_ts = self.get_negative_samples(batch_size_each) #[b,n_neg,d]


        neg_losses = define_loss([t, neg_ts])  #[b,num_neg]

        if param.knowledge == 'rotate':
            # current loss is actually dist
            gm = param.gamma
            pos_loss1 =  torch.squeeze(-torch.log(nn.functional.softplus(gm - pos_loss)))
            neg_losses1 = -torch.log(nn.functional.softplus(neg_losses - gm))
            neg_loss1 = torch.mean(neg_losses1, dim=-1)
            total_loss = (pos_loss1 + neg_loss1) / 2

            pos_loss_other_code = F.logsigmoid(RotatE([h, r],t,self.device)).squeeze(dim=1)
            neg_loss_other_code = F.logsigmoid(RotatE([h, r],neg_ts,self.device)).mean(dim=1)
            positive_sample_loss = - pos_loss_other_code.mean()
            negative_sample_loss = - neg_loss_other_code.mean()

            total_loss = (positive_sample_loss + negative_sample_loss)/2


        else:
            neg_loss = torch.mean(neg_losses, dim=-1)

            target = torch.tensor([-1], dtype=torch.long, device=self.device)
            total_loss = self.criterion(pos_loss,neg_loss,target)
            # total_loss = torch.max(pos_loss - neg_loss + param.margin, 0)


        #return torch.mean(total_loss)
        return total_loss

    def predict(self,h,r,device):
        #(h,r) -> projected t vector
        # h = torch.index_select(
        #     self.entity_embedding_layer,
        #     dim=0,
        #     index=h
        # ).unsqueeze(1)
        #
        # r = torch.index_select(
        #     self.rel_embedding_layer,
        #     dim=0,
        #     index=r
        # ).unsqueeze(1)

        h = self.entity_embedding_layer(h).unsqueeze(1)

        r = self.rel_embedding_layer(r).unsqueeze(1)


        projected_t = project_t([h, r],device)

        return projected_t



class create_alignment_model(nn.Module):
    def __init__(self,other_kg,self_kg,other_lang,self_lang):
        super(create_alignment_model, self).__init__()
        self.other_lang = other_lang 
        self.lang = self_lang

        self.other_kg_entity_embedding = other_kg.entity_embedding_layer
        self.kg_entity_embedding = self_kg.entity_embedding_layer

    def calculate_align_loss(self,input_e0,input_e1):
        '''

        :param input_e0: entity input from other kg
        :param input_e1: entity input for the current kg
        :return:
        '''
    # entity embedding. Don't name it or retrieve by name, otherwise name conflict


        # e0 = torch.index_select(
        #     self.other_kg_entity_embedding,
        #     dim=0,
        #     index=input_e0
        # )


        e0 = self.other_kg_entity_embedding(input_e0)

        # e1 = torch.index_select(
        #     self.kg_entity_embedding,
        #     dim=0,
        #     index=input_e1
        # )
        e1 = self.kg_entity_embedding(input_e1)

        align_loss = torch.mean(l2distance(e0, e1)) # num batch

        return align_loss





def KNN_finder(predictor,input_h_query, input_r_query, embedding_matrix,device):
    """
    kNN finder
    === input: predictor, [input_h_query, input_r_query, embedding_matrix]
    === output:[top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    predicted_t = predictor.predict(input_h_query, input_r_query,device)
    kNN_idx_and_score = find_kNN([predicted_t, embedding_matrix])

    return kNN_idx_and_score


def KNN_finder_vec(input_vec,embedding_matrix,topk = param.k):
    """
    Given a vector, find the kNN entities
    :param num_entity:
    :return:
    """
    kNN_idx_and_score = find_kNN([input_vec, embedding_matrix],topk)

    return kNN_idx_and_score


def find_kNN(t_vec_and_embed_matrix,topk = param.k):
    """
    :param t_vec_and_embed_matrix:
    :return: [top_k_entity_idx, top_k_scores(larger is better)] . top_k_entity_idx shape [1,k].
                e.g., [array([[64931, 13553, 20458]]), array([[-1.008282 , -2.0292854, -3.059666]])]
    """
    predicted_t_vec = torch.squeeze(t_vec_and_embed_matrix[0])  # shape (batch_size=1, 1, dim) -> (dim,)
    embedding_matrix = t_vec_and_embed_matrix[1]
    try:
        distance = torch.norm(torch.sub(embedding_matrix.weight, predicted_t_vec), dim=1)
    except:
        distance = torch.norm(torch.sub(embedding_matrix, predicted_t_vec), dim=1)
    top_k_scores, top_k_t = torch.topk(-distance, k=topk)  # find indices of k largest score. score = neg(distance)
    return [torch.reshape(top_k_t, [1, topk]), torch.reshape(top_k_scores, [1, topk])]  # reshape to one row matrix to fit keras model output



def create_knowledge_model(num_entity, num_relation, device,relation_layer=None):
    """
    model: (h,r,t) -> loss/distance/-score. Main model, used for training embedding.
    predictor: (h,r) -> predicted t vector
    kNN finder: (h,r) -> top k possible t
    :return: model, predictor, kNN_finder
    """
    model = KGFunction(device,num_entity,num_relation)

    return model

def project_t(hr,device):
    if param.knowledge == 'transe':
        return hr[0] + hr[1]
    elif param.knowledge == 'rotate':
        pi = torch.FloatTensor([3.14159265358979323846]).to(device)
        head, relation = hr[0], hr[1]

        re_head, im_head = torch.chunk(head, 2, dim=2)  # input shape: (None, 1, dim)

        embedding_range = torch.FloatTensor([param.rotate_embedding_range()]).to(device)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(embedding_range/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_tail = re_head * re_relation - im_head * im_relation
        im_tail = re_head * im_relation + im_head * re_relation

        predicted_tail = torch.cat([re_tail, im_tail], dim=-1)

        return predicted_tail

def RotatE(hr,tail,device):
    # calculate positive score
    pi = 3.14159265358979323846
    head, relation = hr[0], hr[1]

    re_head, im_head = torch.chunk(head, 2, dim=2)
    re_tail, im_tail = torch.chunk(tail, 2, dim=2)

    # Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = torch.FloatTensor([param.rotate_embedding_range()]).to(device)
    phase_relation = relation/(embedding_range/pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    re_score = re_score - re_tail
    im_score = im_score - im_tail

    score = torch.stack([re_score, im_score], dim=0)
    score = score.norm(dim=0)

    score = param.gamma - score.sum(dim=2)

    return score


def define_loss(t_true_pred):
    t_true = t_true_pred[0]
    t_pred = t_true_pred[1]

    # tf.norm() will result in nan loss when tf.norm([0])
    # USE tf.reduce_mean(tf.square()) INSTEAD!!!
    # return tf.reduce_mean(tf.square(t_true-t_pred), axis=2)  # input shape: (None, 1, dim)
    return torch.norm(t_true - t_pred + 1e-8, dim=2)  # input shape: (None, 1, dim)



def l2distance(a, b):
    # dist = tf.sqrt(tf.reduce_sum(tf.square(a-b), axis=-1))
    dist = torch.norm(a - b + 1e-8, dim=-1)
    return dist



def extract_entity_embedding_matrix(knowledge_model):
    """
    Get the embedding matrix and FLATTEN it to ensure successful feed to keras layer
    :param knowledge_model:
    :return:
    """
    return torch.squeeze(knowledge_model.entity_embedding_layer).reshape([1, -1])


def save_model_structure(model, output_path):
    json_string = model.to_json()
    with open(output_path, 'w') as outfile:
        outfile.write(json_string)



def extend_seed_align_links(kg0, kg1, seed_links,device):
    """
    Self learning using cross-domain similarity scaling (CSLS) metric for kNN search
    :param kg0: supporter kg
    :param kg1: target kg
    :param seed_links: 2-col np array
    :return:
    """
    def cos(v1, v2):
        return F.cosine_similarity(v1,v2,dim=-1)

    csls_links = []

    aligned0 = torch.unique(seed_links[:, 0],return_inverse=False)
    aligned1 = torch.unique(seed_links[:, 1],return_inverse=False)

    k_csls = 3  # how many nodes in neiborhood
    k_temp = param.k
    param.k = k_csls


    embedding_matrix0_reshaped = kg0.model.entity_embedding_layer
    embedding_matrix1_reshaped = kg1.model.entity_embedding_layer

    # change param.k back for link prediction
    param.k = k_temp

    # find kNN for each e0
    # mean neighborhood similarity
    e0_neighborhood = torch.zeros([kg0.num_entity, k_csls],dtype=torch.long).to(device)
    e1_neighborhood = torch.zeros([kg1.num_entity, k_csls],dtype=torch.long).to(device)
    e0_neighborhood_cos = torch.zeros(kg0.num_entity).to(device)
    e1_neighborhood_cos = torch.zeros(kg1.num_entity).to(device)

    # find neighborhood
    for e0 in range(kg0.num_entity):
        top_k_from_kg1 = KNN_finder_vec(
            embedding_matrix0_reshaped[e0, :].reshape([1, -1]), embedding_matrix1_reshaped,k_csls)  # [array(entity), array(score)]
        neighbood = top_k_from_kg1[0]  # list[entity], possible e1
        e0_neighborhood[e0, :] = neighbood
    for e1 in range(kg1.num_entity):
        top_k_from_kg0 = KNN_finder_vec(
            embedding_matrix1_reshaped[e1, :].reshape([1, -1]), embedding_matrix0_reshaped,k_csls)
        neighbood = top_k_from_kg0[0]  # list[entity], possible e0
        e1_neighborhood[e1, :] = neighbood

    # e0_neighborhood = e0_neighborhood.astype(np.int32)
    # e1_neighborhood = e1_neighborhood.astype(np.int32)

    # compute neighborhood similarity
    for e0 in range(kg0.num_entity):
        e0_vec = embedding_matrix0_reshaped[e0]
        e0_neighbors = e0_neighborhood[e0, :]  # e0's neighbor in kg1 domain
        neighbor_cos = [cos(embedding_matrix1_reshaped[nb, :], e0_vec) for nb in e0_neighbors]
        e0_neighborhood_cos[e0] = torch.mean(torch.FloatTensor(neighbor_cos).to(device))  # r_S

    for e1 in range(kg1.num_entity):
        e1_vec = embedding_matrix1_reshaped[e1]
        e1_neighbors = e1_neighborhood[e1, :]  # e0's neighbor in kg1 domain
        neighbor_cos = [cos(embedding_matrix0_reshaped[nb, :], e1_vec) for nb in e1_neighbors]
        e1_neighborhood_cos[e1] = torch.mean(torch.FloatTensor(neighbor_cos).to(device))

    nearest_for_e0 = torch.full((kg0.num_entity,1), fill_value=-2).to(device)  # -2 for not computed, -1 for not found
    nearest_for_e1 = torch.full((kg1.num_entity,1), fill_value=-2).to(device)

    for true_e0 in range(kg0.num_entity):
        if true_e0 not in aligned0:
            e0_neighbors = e0_neighborhood[true_e0, :]  # e0's neighbor in kg1 domain
            nearest_e1 = torch.LongTensor([-1]).to(device)
            nearest_e1_csls = torch.FloatTensor([-np.inf]).to(device)
            for e1 in e0_neighbors.tolist():
                if e1 not in aligned1:
                    # rT(Wx_s) is the same for all e1 in e0's neighborhood
                    csls = 2 * cos(embedding_matrix0_reshaped[true_e0, :], embedding_matrix1_reshaped[e1, :]) - \
                           e1_neighborhood_cos[e1]
                    if csls > nearest_e1_csls:
                        nearest_e1 = e1
            nearest_for_e0[true_e0] = nearest_e1

            # check if they are mutual neighbors
            if nearest_for_e0[true_e0] != torch.LongTensor([-1]).to(device):
                e1 = nearest_for_e0[true_e0]
                if nearest_for_e1[e1] == torch.LongTensor([-2]).to(device):  # e1's nearest number not computed yet. compute it now
                    e1_neighbors = e1_neighborhood[e1[0], :]  # e0's neighbor in kg1 domain
                    nearest_e0 = torch.LongTensor([-1]).to(device)
                    nearest_e0_csls = torch.FloatTensor([-np.inf]).to(device)
                    for e0 in e1_neighbors:
                        if e0 not in aligned0:
                            # rT(Wx_s) is the same for all e1 in e0's neighborhood
                            csls = 2 * cos(embedding_matrix1_reshaped[e1, :], embedding_matrix0_reshaped[e0, :]) - \
                                   e0_neighborhood_cos[e0]
                            if csls > nearest_e0_csls:
                                nearest_e0 = e0
                                nearest_e0_csls = csls
                    nearest_for_e1[e1] = nearest_e0

                if nearest_for_e1[e1] == true_e0:
                    # mutual e1_neighbors
                    csls_links.append([true_e0, e1])


    csls_links = torch.LongTensor(csls_links).to(device)

    return csls_links
