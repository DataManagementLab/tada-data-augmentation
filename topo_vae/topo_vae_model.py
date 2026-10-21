import pytorch_lightning as pl
import os
import numpy as np
import torch
import torch.nn as nn


class TopologicallyRegularizedVariationalAutoencoder(pl.LightningModule):
    """Topologically regularized variational autoencoder."""

    def __init__(self, **kwargs):
        """Topologically Regularized variational Autoencoder.

        Args:
            kwargs: Keywords defined in clean_interface class
        """
        super().__init__()
        latent_dim = kwargs["latent_dim"]
        hidden_dim = kwargs["hidden_dim"]

        self.beta = kwargs["beta"]

        self.cat_cols = kwargs["cat_cols"]
        self.reg_cols = kwargs["reg_cols"]
        self.num_cat_vals_per_col = kwargs["num_cat_vals_per_col"]

        self.embeddings = nn.ModuleDict()

        total_emb_dim = 0

        if len(self.cat_cols) > 0:
            for i in range(len(self.num_cat_vals_per_col)):
                emb_dim = 10

                self.embeddings[str(i)] = nn.Embedding(
                    self.num_cat_vals_per_col[i], emb_dim
                )

                total_emb_dim += emb_dim

        input_dim = len(self.reg_cols) + total_emb_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.numerical_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.reg_cols)),
        )

        if len(self.cat_cols) > 0:
            self.cat_heads = nn.ModuleDict()

            for i in range(len(self.num_cat_vals_per_col)):
                self.cat_heads[str(i)] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.num_cat_vals_per_col[i]),
                )

        self.topo_sig = TopologicalSignatureDistance(match_edges="symmetric")
        self.bce_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.topo_n_batches = kwargs["topo_every_n_batches"]
        self.topo_start_epoch = kwargs["topo_start_epoch"]

        self.learning_rate = kwargs["lr"]
        self.topo_weight = kwargs["topo_weight"]

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.01)

    def encode(self, x):

        if len(self.cat_cols) > 0:
            x_cat = x[:, self.cat_cols]
            x_num = x[:, self.reg_cols]

            emb_list = []
            for col, emb_layer in self.embeddings.items():
                emb = emb_layer(x_cat[:, int(col)].int())

                emb_list.append(emb)

            x = torch.cat([x_num] + emb_list, dim=1)

        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        dec_1 = self.decoder(z)

        num_out = self.numerical_decoder(dec_1)

        if len(self.cat_cols) > 0:
            cat_out = torch.zeros((z.shape[0], len(self.cat_cols)))

            for col, head in self.cat_heads.items():
                probs = torch.softmax(head(dec_1), dim=1)
                sampled = torch.argmax(probs, dim=1)
                cat_out[:, int(col)] = sampled.view(-1)

            com_vec = torch.zeros((z.shape[0], len(self.reg_cols) + len(self.cat_cols)))
            com_vec[:, self.reg_cols] = num_out
            com_vec[:, self.cat_cols] = cat_out
        else:
            com_vec = num_out
        return com_vec

    def decode_sel(self, z):
        dec_1 = self.decoder(z)

        num_out = self.numerical_decoder(dec_1)
        probs = torch.ones((z.shape[0], 1))
        if len(self.cat_cols) > 0:
            cat_out = torch.zeros((z.shape[0], len(self.cat_cols)))

            for col, head in self.cat_heads.items():
                probs = torch.softmax(head(dec_1), dim=1)
                sampled = torch.argmax(probs, dim=1)
                cat_out[:, int(col)] = sampled.view(-1)

            com_vec = torch.zeros((z.shape[0], len(self.reg_cols) + len(self.cat_cols)))
            com_vec[:, self.reg_cols] = num_out
            com_vec[:, self.cat_cols] = cat_out
        else:
            com_vec = num_out
        return com_vec, probs

    def decode_train(self, z):
        dec_1 = self.decoder(z)

        num_out = self.numerical_decoder(dec_1)

        if len(self.cat_cols) > 0:
            cat_out = {}

            for col, head in self.cat_heads.items():
                cat_out[col] = head(dec_1)

            return num_out, cat_out
        else:
            return num_out, None

    def forward(self, x):
        mu, logvar = self.encode(x)
        embedding = self.reparameterize(mu, logvar)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, x_distances, y = train_batch

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        num_out, cat_out = self.decode_train(z)

        # topology loss
        x_distances = x_distances / x_distances.max()

        latent_distances = torch.cdist(mu, mu, p=2)
        latent_distances = latent_distances / latent_distances.max()

        x_num = x[:, self.reg_cols]
        bce_loss = self.bce_loss(num_out, x_num)

        if len(self.cat_cols) > 0:
            x_cat = x[:, self.cat_cols]

            cat_loss = torch.tensor(0.0, requires_grad=True)

            for col in cat_out:
                cat_loss = cat_loss + self.cross_entropy(
                    cat_out[col],
                    x_cat[:, int(col)].long(),
                )
            cat_loss = cat_loss / len(cat_out)

            cat_out_ = torch.zeros((z.shape[0], len(self.cat_cols)))

            for i in range(len(cat_out)):
                cat_out_[:, i] = torch.argmax(cat_out[str(i)], dim=1)

        if self.topo_start_epoch < self.current_epoch:
            if batch_idx % self.topo_n_batches == 0:
                loss_topo, topo_loss_components = self.topo_sig(
                    x_distances, latent_distances
                )
                loss_topo = loss_topo / float(x.shape[0])
            else:
                loss_topo = torch.tensor(0.0, requires_grad=True)
        else:
            loss_topo = torch.tensor(0.0, requires_grad=True)

        recon_loss = (bce_loss + cat_loss) / 2 if len(self.cat_cols) > 0 else bce_loss

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        loss = recon_loss + self.beta * kl_loss + self.topo_weight * loss_topo
        return loss

    def save_model(self, path, filename):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
        }
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(checkpoint, os.path.join(path, filename))

    def load_model(self, path, filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(
            os.path.join(path, filename),
        )
        self.load_state_dict(checkpoint["state_dict"])

    def save_pretrained(self, path):
        state_dict = self.state_dict()

        state_dict = {
            name: value
            for name, value in state_dict.items()
            if not (
                name.startswith("embeddings.")
                or name.startswith("cat_heads.")
                or name.startswith("encoder.0.")
                or name.startswith("numerical_decoder.2.")
            )
        }

        torch.save(state_dict, path)

    def load_pretrained(self, path):
        state_dict = torch.load(path, weights_only=True)

        return self.load_state_dict(state_dict, strict=False)


########################## TopoAE topological loss ##########################
##########################     DO NOT TOUCH        ##########################


class UnionFind:
    """
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    """

    __slots__ = ("_parent",)

    def __init__(self, n_vertices):
        """
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        """
        # A plain Python list of Python ints is faster to read/write in
        # a tight Python loop than a numpy array: every access to a
        # numpy array element allocates a new numpy scalar object,
        # whereas list access/mutation is a direct C-level operation on
        # a native int.
        self._parent = list(range(n_vertices))

    def find(self, u):
        """
        Finds and returns the parent of u with respect to the hierarchy.
        Applies full path compression, done iteratively rather than
        recursively -- this avoids Python function-call overhead per
        step and removes any risk of hitting Python's recursion limit
        on large inputs (a long union-find chain used to be able to
        raise RecursionError).
        """
        parent = self._parent

        root = u
        while parent[root] != root:
            root = parent[root]

        # Second pass: point every visited node directly at the root.
        while parent[u] != root:
            parent[u], u = root, parent[u]

        return root

    def merge(self, u, v):
        """
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        """
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def merge_roots(self, root_u, root_v):
        """
        Merges the component rooted at root_u into the component rooted
        at root_v, given that both roots are already known (e.g. from a
        prior call to find()). This is a fast path that skips the
        redundant find() calls merge() would otherwise perform when the
        caller has already computed both roots -- useful in hot loops
        like PersistentHomologyCalculation, where find() is called on
        u and v right before merging anyway.
        """
        self._parent[root_u] = root_v

    def roots(self):
        """
        Generator expression for returning roots, i.e. components that
        are their own parents.
        """
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __call__(self, matrix):
        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        # k=1 excludes the diagonal. Self-edges (u == v) can never join
        # the MST -- find(u) always equals find(v) -- so there's no
        # reason to sort or iterate over them at all.
        triu_row, triu_col = np.triu_indices(n_vertices, k=1)
        edge_weights = matrix[triu_row, triu_col]
        edge_order = np.argsort(edge_weights, kind="stable")

        # Do the reordering once, vectorized, with numpy; then drop to
        # plain Python lists. Kruskal's algorithm is inherently
        # sequential (each step depends on the union-find state left by
        # the previous one) so the main loop itself can't be
        # vectorized away -- but indexing numpy arrays element-by-element
        # inside that loop is noticeably slower than indexing plain
        # Python lists, since every numpy element access allocates a
        # scalar object.
        us = triu_row[edge_order].tolist()
        vs = triu_col[edge_order].tolist()

        # A minimum spanning tree over n_vertices always has exactly
        # n_vertices - 1 edges. Preallocating avoids the overhead of
        # repeated list.append() + the final list -> np.array conversion,
        # and guarantees a consistent (k, 2) shape even in the n=1
        # degenerate case (the original code returned a 1-D empty array
        # in that case, which downstream code indexing pairs[:, 0] would
        # not actually be able to handle).
        persistence_pairs = np.empty((max(n_vertices - 1, 0), 2), dtype=np.int64)
        n_pairs = 0

        find = uf.find
        merge_roots = uf.merge_roots
        target_pairs = n_vertices - 1

        for u, v in zip(us, vs):
            younger_component = find(u)
            older_component = find(v)

            # Not an edge of the MST, so skip it.
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                # Equivalent to uf.merge(v, u), but reuses the roots we
                # already computed above instead of recomputing find(u)
                # and find(v) a second time inside merge().
                merge_roots(older_component, younger_component)
            else:
                merge_roots(younger_component, older_component)

            if u < v:
                persistence_pairs[n_pairs] = (u, v)
            else:
                persistence_pairs[n_pairs] = (v, u)
            n_pairs += 1

            # Once every vertex has been merged into a single component,
            # every remaining edge is guaranteed to be a same-component
            # no-op (Kruskal's stopping condition), so we can stop early
            # instead of burning through the rest of the sorted edges.
            if n_pairs == target_pairs:
                break

        # Return empty cycles component
        return persistence_pairs[:n_pairs], np.array([])


class AlephPersistenHomologyCalculation:
    def __init__(self, compute_cycles, sort_selected):
        """Calculate persistent homology using aleph.

        Args:
            compute_cycles: Whether to compute cycles
            sort_selected: Whether to sort the selected pairs using the
                distance matrix (such that they are in the order of the
                filteration)
        """
        self.compute_cycles = compute_cycles
        self.sort_selected = sort_selected

    def __call__(self, distance_matrix):
        """Do PH calculation.

        Args:
            distance_matrix: numpy array of distances

        Returns: tuple(edge_featues, cycle_features)
        """
        import aleph

        if self.compute_cycles:
            pairs_0, pairs_1 = aleph.vietoris_rips_from_matrix_2d(distance_matrix)
            pairs_0 = np.array(pairs_0)
            pairs_1 = np.array(pairs_1)
        else:
            pairs_0 = aleph.vietoris_rips_from_matrix_1d(distance_matrix)
            pairs_0 = np.array(pairs_0)
            # Return empty cycles component
            pairs_1 = np.array([])

        if self.sort_selected:
            selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]
            indices_0 = np.argsort(selected_distances)
            pairs_0 = pairs_0[indices_0]
            if self.compute_cycles:
                cycle_creation_times = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
                cycle_destruction_times = distance_matrix[
                    (pairs_1[:, 2], pairs_1[:, 3])
                ]
                cycle_persistences = cycle_destruction_times - cycle_creation_times
                # First sort by destruction time and then by persistence of the
                # create cycles in order to recover original filtration order.
                indices_1 = np.lexsort((cycle_destruction_times, cycle_persistences))
                pairs_1 = pairs_1[indices_1]

        return pairs_0, pairs_1


class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False, match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        if use_cycles:
            use_aleph = True
        else:
            if not sort_selected and match_edges is None:
                use_aleph = True
            else:
                use_aleph = False

        if use_aleph:
            print("Using aleph to compute signatures")
            self.signature_calculator = AlephPersistenHomologyCalculation(
                compute_cycles=use_cycles, sort_selected=sort_selected
            )
        else:
            print("Using python to compute signatures")
            self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat((selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2) ** 2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)

        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            "metrics.matched_pairs_0D": self._count_matching_pairs(pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components["metrics.matched_pairs_1D"] = (
                self._count_matching_pairs(pairs1[1], pairs2[1])
            )
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components["metrics.non_zero_cycles_1"] = nonzero_cycles_1
            distance_components["metrics.non_zero_cycles_2"] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == "symmetric":
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components["metrics.distance1-2"] = distance1_2
            distance_components["metrics.distance2-1"] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == "random":
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat(
                [
                    torch.randperm(n_instances)[:, None],
                    torch.randperm(n_instances)[:, None],
                ],
                dim=1,
            )
            pairs2 = torch.cat(
                [
                    torch.randperm(n_instances)[:, None],
                    torch.randperm(n_instances)[:, None],
                ],
                dim=1,
            )

            sig1_1 = self._select_distances_from_pairs(distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components["metrics.distance1-2"] = distance1_2
            distance_components["metrics.distance2-1"] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components
