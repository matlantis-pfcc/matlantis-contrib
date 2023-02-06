template_options = """
ITEM	OPTIONS

replace		true
#workdir		./work
#debug		false
field		{field}
density		{density}
ntotal		{ntotal}
build_dir	{build_dir}
build_replace	true
pdb		true
pdb_compress	false
emc_execute {emc_execute}	
prefix		{lammps_prefix}
project		{project}
field_error	false   # Ignore errors of force field assignment.
#charge		false
#emc_test	true
#expert		true
#depth		{ring_depth}
emc_depth 6

ITEM	END
"""

template_groups = """
# Groups

ITEM	GROUPS

{groups}

ITEM	END
"""

template_clusters = """
# Clusters

ITEM	CLUSTERS

{clusters}

ITEM	END
"""


template = template_options + template_groups + template_clusters
