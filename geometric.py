from typing import Dict, List, Tuple

import torch as th
from torch import Tensor

from torch_geometric.data import Batch, Data


def collate(
	obs: Dict[str, Tensor]
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, int]:
	"""
	Preprocess the observation if needed and extract features.

	:param obs: Observation
	:param features_extractor: The features extractor to use.
	:return: The extracted features
	"""

	datalist: List[Data] = [
		Data(
			x=obs["nodes"][i][~obs["node_padding_mask"][i].bool()],
			edge_index=obs["edges"][i][
				~obs["edge_padding_mask"][i].bool()
			].T.long(),
			edge_attr=th.zeros(0),
		)
		for i in range(obs["nodes"].shape[0])
	]

	batch = Batch.from_data_list(datalist)

	return batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.num_graphs