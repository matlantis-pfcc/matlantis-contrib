#BINDIR=$(HOME)/bin/
BINDIR=${PWD}
#for g++ compiler on all platforms
CXX=g++

#for MPI: limited implementation at this point
MPICXX=mpiCC -DATAT_MPI 

all:
	bash ./foolproof ${BINDIR}   "${CXX}"
	${MAKE} -C src clean
	${MAKE} -C src          "CXX=${CXX}"
    
genstr:
	bash ./foolproof ${BINDIR}   "${CXX}"
	${MAKE} -C src clean
	${MAKE} -C src genstr   "CXX=${CXX}"

mpi:
	bash ./foolproof ${BINDIR}   "${MPICXX}"
	${MAKE} -C src clean
	${MAKE} -C src          "CXX=${MPICXX}"  mpi

force:
	${MAKE} -C src          "CXX=${CXX}"

clean:
	${MAKE} -C src clean

install:
	echo set atatdir="$(PWD)" > $(HOME)/.atat.rc
	${MAKE} -C src          BINDIR=${BINDIR} install
	${MAKE} -C glue/jobctrl BINDIR=${BINDIR} install
	${MAKE} -C glue/vasp    BINDIR=${BINDIR} install

uninstall:
	rm -f `cat src/safecpdir/touninstall.txt glue/jobctrl/safecpdir/touninstall.txt glue/vasp/safecpdir/touninstall.txt`
	rm -i ${HOME}/.atat.rc ${HOME}/.ezvasp ${HOME}/.machines.rc
