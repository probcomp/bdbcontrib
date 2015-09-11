default-target: all
default-target: .PHONY
.PHONY:

all: .PHONY
all: build/satellites/satellites.html

clean: .PHONY
	-rm -rf build

build/stamp:
	mkdir -p build
	touch build/stamp

build/satellites/satellites.html: satellites/satellites.bql build/stamp
	rm -rf build/satellites.tmp && \
	cd satellites && \
	(echo ".hook ../../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../../bdbcontrib/contrib_math.py" && \
	 echo ".readtohtml satellites.bql ../build/satellites.tmp/") \
	| PYTHONPATH=../../:$$PYTHONPATH bayeslite -m --seed 0 && \
	rm -rf build/satellites && \
	mv build/satellites.tmp build/satellites

build/satellites.bdb: satellites/data/satellites.utf8.csv build/stamp
	rm -f $@ && \
	(echo ".csv satellites satellites/data/satellites.utf8.csv" && \
	 echo ".nullify satellites NaN" && \
	 echo "CREATE DEFAULT GENERATOR satellites_cc FOR satellites \
	       USING crosscat (GUESS(*), name IGNORE);" && \
	 echo "INITIALIZE 16 MODELS FOR satellites;" && \
	 echo "ANALYZE satellites FOR 4 MINUTES WAIT;") \
	| bayeslite -j0 --seed 0 $@.tmp && mv -f $@.tmp $@

build/malawi/malawi.html: malawi/malawi.bql build/stamp
	rm -rf build/malawi.tmp && \
	# Set the working directory because the .bql script depends on it
	cd malawi && \
	(echo ".hook ../../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../../bdbcontrib/contrib_math.py" && \
	 echo ".readtohtml malawi.bql ../build/malawi.tmp/") \
	| PYTHONPATH=../../:$$PYTHONPATH bayeslite -m --seed 0 && \
	rm -rf build/malawi && \
	mv build/malawi.tmp build/malawi

heatmap: build/satellites.bdb .PHONY
	(echo ".hook ../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../bdbcontrib/contrib_math.py" && \
	 echo ".heatmap 'ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM satellites_cc;'") \
	| PYTHONPATH=../:$$PYTHONPATH bayeslite build/satellites.bdb --seed 0
