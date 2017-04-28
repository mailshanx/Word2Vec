#!/usr/bin/env bash


if (conda env list | grep $1) 
then
	echo ">>> updating environment"
	conda env update -f environment.yml
else
	echo ">>> Setting up new Conda environment $1"
	conda env create -f environment.yml --name $1
	echo ">>> New conda env created. Activate with: \n source activate $1"
fi

