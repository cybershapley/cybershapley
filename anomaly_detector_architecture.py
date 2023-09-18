import torch.nn as nn
import torch


class ad_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, p_dropout, bidirectional):#,
        super(ad_model, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                           dropout=p_dropout,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, logs):
        D = 2 if self.bidirectional else 1
        num_layers=D*self.num_layers
        batch_size = logs.shape[0]
        hidden_size = self.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_size)
        c0 = torch.zeros(num_layers, batch_size, hidden_size)
        out, _ = self.lstm(logs.float(), (h0.float(), c0.float()))
        out = self.fc(out)#[:, -1, :])
        return out



#
# class DataHandler():
#
#     def __init__(self, data_path):
#         self.events_df = pd.read_csv(data_path)
#         self.sequence_len = 10 # set seq len:
#         event_ls = [
#             "EVENT_EXECUTE",
#             "EVENT_FORK",
#             "EVENT_MODIFY_FILE_ATTRIBUTES",
#             "EVENT_READ",
#             "EVENT_RECVFROM",
#             "EVENT_SENDTO",
#             "EVENT_WRITE"
#         ]
#         host_ls = [
#             '83C8ED1F-5045-DBCD-B39F-918F0DF4F851'
#         ]
#         event2code = {k: i for i, k in enumerate(self.events_df['type'].unique())}  # fix this !!!!!!!
#
#         self.events_df['encoded_type'] = self.events_df['type'].apply(lambda x: event2code[x])#.values
#
#         #for now we have only one host - if we will have more on the future use -
#         # host2code = {k: i for i, k in enumerate(host_ls)}
#         # full_df['hostId'] = full_df['hostId'].apply(lambda x: host2code[x])
#
#
#
#
#     def make_seq(self, seq_ind):
#         relevant_df = self.events_df.iloc[seq_ind:seq_ind + sequence_len + 1]
#
#         eid_list = relevant_df['eid'].values
#         sid_list = relevant_df['subject'].values
#         oid_list = relevant_df['object'].values
#         etype_ls = relevant_df['encoded_type'].values
#         # ttps = relevant_df["techniques"].values
#
#         labels = etype_ls[1:]  # the labels are the next event types
#         seq = [] # sequence that include event type, present and next subject and object
#
#         for event_ind in range(len(relevant_df)-1): #-1 because we won't take the 11'th event
#             log = []
#             log.append(etype_ls[event_ind])
#
#             # for present subject
#             same_subjects = [int(x == sid_list[event_ind]) for x in sid_list[:-1]]
#             log.extend(same_subjects)
#
#             # for next subject
#             same_subjects_next = [int(x == sid_list[event_ind + 1]) for x in sid_list[:-1]]
#             log.extend(same_subjects_next)
#
#             # for present object
#             same_objects = [int(x == oid_list[event_ind]) for x in oid_list[:-1]]
#             log.extend(same_objects)
#
#             # for next object
#             same_objects_next = [int(x == oid_list[event_ind + 1]) for x in oid_list[:-1]]
#             log.extend(same_objects_next)
#
#             seq.append(log)
#
#         return seq, labels
#
#
#
#
#     # def make_seqs(df):
#     #     res = []  # {'Sequentials': []}
#     #     labels = []
#     #     # uniuqe_host = df["hostId"].unique().tolist()
#     #     event_ids = []
#     #     # uniuqe_host= [i[0] for i in uniuqe_host]
#     #     # uniuqe_host = list(set(uniuqe_host))
#     #     # print(i)
#     #     events_df = df#df[df.hostId == i]
#     #     etype_ls = events_df[encoded_type]
#     #     # print(my_list)
#     #     eid_list = events_df['eid'].values
#     #     sid_list = events_df['subject'].values
#     #     oid_list = events_df['object'].values
#     #     for seq_ind in range(len(etype_ls) - sequence_len):
#     #
#     #         #             node_dict =dict()
#     #         unique_subjects = np.unique(sid_list[seq_ind:seq_ind + sequence_len + 1])
#     #         #             same_subjects = [int(x == rec) for x in sid_list[seq_ind:seq_ind+window+1]]
#     #
#     #         unique_objects = np.unique(oid_list[seq_ind:seq_ind + sequence_len + 1])
#     #         #             same_objects = [int(x == rec) for x in oid_list[seq_ind:seq_ind+window+1]]
#     #
#     #         event_ids.append(eid_list[seq_ind:seq_ind + sequence_len].reshape(-1, 1))
#     #         labels.append(etype_ls[seq_ind + 1:seq_ind + 1 + sequence_len])
#     #
#     #         # create input for each event in 1 sequence
#     #         input_list = []
#     #         for event_ind in range(sequence_len):
#     #             event_tuple = []
#     #             # for (present) event type
#     #             event_tuple.append(etype_ls[seq_ind + event_ind])
#     #
#     #             # for present subject
#     #             same_subjects = [int(x == sid_list[seq_ind + event_ind]) for x in
#     #                              sid_list[seq_ind:seq_ind + sequence_len]]
#     #             event_tuple.extend(same_subjects)
#     #
#     #             # for next subject
#     #             same_subjects_next = [int(x == sid_list[seq_ind + event_ind + 1]) for x in
#     #                                   sid_list[seq_ind + 1:seq_ind + 1 + sequence_len]]
#     #             event_tuple.extend(same_subjects_next)
#     #
#     #             # for present object
#     #             same_objects = [int(x == oid_list[seq_ind + event_ind]) for x in
#     #                             oid_list[seq_ind:seq_ind + sequence_len]]
#     #             event_tuple.extend(same_objects)
#     #
#     #             # for next object
#     #             same_objects_next = [int(x == oid_list[seq_ind + event_ind + 1]) for x in
#     #                                  oid_list[seq_ind + 1:seq_ind + 1 + sequence_len]]
#     #             event_tuple.extend(same_objects_next)
#     #
#     #             input_list.append(
#     #                 event_tuple)  # imput sequence that include event type, present and next subject and object
#     #
#     #         res.append(input_list)
#     #
#     #     return res, labels, event_ids  # label - the 11'th event
#
#     def generate_graph_for_sequence(self, seq_ind):
#
#         node_label_dict = {}
#         edge_label_dict = {}
#         edges_ls = []
#         nodes_ls = []
#         edge_to_event_ind = {}
#         relevant_df = self.events_df.iloc[seq_ind:seq_ind + sequence_len]
#         for ind, row in relevant_df.iterrows():
#             e_type = row["type"]
#             eid = row["eid"]
#             subject_id = row["subject"]
#             object_id = row["object"]
#             ttp = row["techniques"]
#
#             if e_type in ["EVENT_EXECUTE", "EVENT_MODIFY_FILE_ATTRIBUTES", "EVENT_SENDTO"]:
#                 direction_object_to_subject = False
#             elif e_type in ["EVENT_RECVFROM", "EVENT_WRITE"]:
#                 direction_object_to_subject = True
#             else:
#                 raise Exception("Unfamiliar event type")
#             e_type = e_type[6:]
#             node_label_dict[object_id] = "object"
#             node_label_dict[subject_id] = "subject"
#             nodes_ls.append(object_id)
#             nodes_ls.append(subject_id)
#
#             edge = (object_id, subject_id) if direction_object_to_subject else (subject_id, object_id)
#             edges_ls.append(edge)
#             time_stamp = ind - seq_ind
#             edge_to_event_ind[edge] = time_stamp
#             edge_label_dict[edge] = f"e{time_stamp}, {e_type}, {ttp}"
#
#         g = nx.DiGraph()  # Creating Directed Graph #MultiDiGraph
#         # adding nodes and vertices
#         g.add_nodes_from(nodes_ls)
#         g.add_edges_from([(edge[0], edge[1]) for edge in edges_ls])
#         sorted_components = sorted([g.subgraph(subg) for subg in nx.weakly_connected_components(g)], key=len, reverse=True)
#
#         return sorted_components, edge_to_event_ind