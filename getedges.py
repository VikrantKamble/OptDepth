import numpy as np

def unequalhist(a, nobj):
	sort_a = np.sort(a)

	edges = []
	cStep = 0
	while(cStep < len(a)):
		edges.append(sort_a[cStep])
		cStep += nobj
	return edges
