default-target: all
default-target: .PHONY
.PHONY:

all: .PHONY
all: malawi
all: satellites

clean: .PHONY
	-rm -rf build

build/stamp:
	mkdir -p build
	touch build/stamp

satellites: .PHONY
satellites: build/satellites/satellites.html

build/satellites/satellites.html: satellites/satellites.bql build/stamp
	rm -rf build/satellites.tmp && \
	(echo ".hook ../../hooks/hook_utils.py" && \
	 echo ".hook ../../hooks/hook_plots.py" && \
	 echo ".hook ../../hooks/hook_maths.py" && \
	 echo ".readtohtml satellites.bql ../build/satellites.tmp/") \
	| (cd satellites && \
	   BAYESDB_WIZARD_MODE=1 PYTHONPATH=../../:$$PYTHONPATH \
	   bayeslite -j1 -m --seed 0) && \
	rm -rf build/satellites && \
	mv build/satellites.tmp build/satellites

build/satellites.bdb: satellites/satellites.csv build/stamp
	rm -f $@ && \
	(echo ".hook ../hooks/hook_utils.py" && \
	 echo ".csv satellites satellites/satellites.csv" && \
	 echo ".nullify satellites NaN" && \
	 echo "CREATE DEFAULT GENERATOR satellites_cc FOR satellites \
	       USING crosscat (GUESS(*), name IGNORE);" && \
	 echo "INITIALIZE 16 MODELS FOR satellites;" && \
	 echo "ANALYZE satellites FOR 4 MINUTES WAIT;") \
	| BAYESDB_WIZARD_MODE=1 PYTHONPATH=../:$$PYTHONPATH \
	  bayeslite -j0 --seed 0 $@.tmp && mv -f $@.tmp $@

# TODO How to teach Make to rebuild all the bdbs if they're not there?
# Do I even want to, for such a big build?
build/satellites-probes-3200m-30i.pkl: build/satellites-bdbs-3200m-30i/*i.bdb
	satellites/probe.py --outfile $@ \
	 --n_models 1 100 200 300 400 --n_replications 8 \
	 build/satellites-bdbs-3200m-30i/*i.bdb

build/satellites-probes-320m-300i.pkl: build/satellites-bdbs-320m-300i/*i.bdb
	satellites/probe.py --outfile $@ \
	 --n_models 1 10 20 30 40 --n_replications 8 \
	 build/satellites-bdbs-320m-300i/*i.bdb

malawi: .PHONY
malawi: build/malawi/malawi.html

build/malawi/malawi.html: malawi/malawi.bql build/stamp
	rm -rf build/malawi.tmp && \
	(echo ".hook ../../hooks/hook_utils.py" && \
	 echo ".hook ../../hooks/hook_plots.py" && \
	 echo ".hook ../../hooks/hook_maths.py" && \
	 echo ".readtohtml malawi.bql ../build/malawi.tmp/") \
	| (cd malawi && \
	   BAYESDB_WIZARD_MODE=1 PYTHONPATH=../../:$$PYTHONPATH \
	   bayeslite -j1 -m --seed 0) && \
	rm -rf build/malawi && \
	mv build/malawi.tmp build/malawi

heatmap: build/satellites.bdb .PHONY
	(echo ".hook ../hooks/hook_utils.py" && \
	 echo ".hook ../hooks/hook_plots.py" && \
	 echo ".hook ../hooks/hook_maths.py" && \
	 echo ".heatmap 'ESTIMATE DEPENDENCE PROBABILITY FROM PAIRWISE COLUMNS OF satellites_cc;'") \
	| PYTHONPATH=../:$$PYTHONPATH \
	  bayeslite -j1 --seed 0 build/satellites.bdb
