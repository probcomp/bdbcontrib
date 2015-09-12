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
	(echo ".hook ../../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../../bdbcontrib/contrib_math.py" && \
	 echo ".readtohtml satellites.bql ../build/satellites.tmp/") \
	| (cd satellites && \
	    PYTHONPATH=../../:$$PYTHONPATH bayeslite -j1 -m --seed 0) && \
	rm -rf build/satellites && \
	mv build/satellites.tmp build/satellites

build/satellites.bdb: satellites/data/satellites.csv build/stamp
	rm -f $@ && \
	(echo ".hook ../bdbcontrib/contrib_util.py" && \
	 echo ".csv satellites satellites/data/satellites.csv" && \
	 echo ".nullify satellites NaN" && \
	 echo "CREATE DEFAULT GENERATOR satellites_cc FOR satellites \
	       USING crosscat (GUESS(*), name IGNORE);" && \
	 echo "INITIALIZE 16 MODELS FOR satellites;" && \
	 echo "ANALYZE satellites FOR 4 MINUTES WAIT;") \
	| PYTHONPATH=../:$$PYTHONPATH bayeslite -j0 --seed 0 $@.tmp && mv -f $@.tmp $@

malawi: .PHONY
malawi: build/malawi/malawi.html

build/malawi/malawi.html: malawi/malawi.bql build/stamp
	rm -rf build/malawi.tmp && \
	(echo ".hook ../../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../../bdbcontrib/contrib_math.py" && \
	 echo ".readtohtml malawi.bql ../build/malawi.tmp/") \
	| (cd malawi && \
	    PYTHONPATH=../../:$$PYTHONPATH bayeslite -j1 -m --seed 0) && \
	rm -rf build/malawi && \
	mv build/malawi.tmp build/malawi

heatmap: build/satellites.bdb .PHONY
	(echo ".hook ../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../bdbcontrib/contrib_math.py" && \
	 echo ".heatmap 'ESTIMATE DEPENDENCE PROBABILITY FROM PAIRWISE COLUMNS OF satellites_cc;'") \
	| PYTHONPATH=../:$$PYTHONPATH bayeslite -j1 --seed 0 build/satellites.bdb
