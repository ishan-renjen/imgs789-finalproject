from sklearn_extra.cluster import CLARA
import numpy as np

class UnknownClassifier:
    def __init__(self, class_to_superclass, memory_per_class=20, k=15):
        self.memory = SemanticReplayMemory(class_to_superclass, memory_per_class)
        self.memory_per_class = memory_per_class
        self.k = k
        self.class_to_superclass = class_to_superclass
        self.prototype_bank = {}
        self.support_embeddings = None
        self.support_labels = None

    def update(self, embeddings, labels):
        self.memory.add(embeddings, labels)
        self.memory.trim()
        self.rebuild_prototypes()

    def set_support_set(self, embeddings, labels):
        self.support_emb = embeddings
        self.support_labels = labels

    def rebuild_prototypes(self):
        prototype_data = dict()

        for class_id in self.memory.seen_classes():
            embeddings = self.memory.get_class_embeddings(class_id)

            if embeddings is None:
                continue

            prototype_data.setdefault(class_id, []).append(embeddings)

        if self.support_embeddings is not None:
            for class_id in np.unique(self.support_labels):
                class_support_embeddings = self.support_embeddings[self.support_labels == int(class_id)]

                if len(class_support_embeddings) == 0:
                    continue

                prototype_data.setdefault(class_id, []).append(embeddings)    

        prototypes = dict()

        for class_id, class_embeddings in prototype_data.items():
            class_embeddings = np.concatenate(class_embeddings, axis=0)
            class_embeddings = (class_embeddings / (np.linalg.norm(class_embeddings, axis=1, keepdims=True)))

            k_eff = min(k, len(class_embeddings))
            model = CLARA(n_clusters=k_eff, metric="cosine", random_state=seed)
            model.fit(support_2d)
            prototypes[class_to_superclass.get(class_id)] = model.cluster_centers_

        self.prototype_bank = prototypes

    def predict(self, embeddings):
        # returns class prediction, superclass prediction, confidence/distance
        pass