#####################################
# name: Makefile
# author: Dylan Morris <dylan@dylanhmorris.com>
#
# Makefile to generate analyses
# for project
####################################

#####################################
# Directory structure
####################################

default: all

SRC = src
OUT_DIR = out
DATA_DIR = dat
RAW_DIR = $(DATA_DIR)/raw
CLEANED_DIR = $(DATA_DIR)/cleaned
CHAIN_DIR = $(OUT_DIR)/chains
FIG_DIR = $(OUT_DIR)/figures
TAB_DIR = $(OUT_DIR)/tables

#####################################
# Expected shell settings
#
# Check these vs your local
# machine setup if you are having
# difficulty reproducing the
# analysis
#####################################

MKDIR := @mkdir -p
RM := rm -f
RMDIR := rmdir
PIP := pip

#####################################
# Outputs
####################################
CLEANED_DATA = $(CLEANED_DIR)/mpx-data.tsv

DEFAULT_CHAINS = titers.pickle halflives-liquid.pickle halflives-surface.pickle
DEFAULT_CHAIN_PATHS = $(addprefix $(CHAIN_DIR)/, $(DEFAULT_CHAINS))

ADDITIONAL_CHAINS = halflives-liquid-hier.pickle

ALL_CHAINS = $(DEFAULT_CHAINS) $(ADDITIONAL_CHAINS)
ALL_CHAIN_PATHS = $(addprefix $(CHAIN_DIR)/, $(ALL_CHAINS))

FIGURE_NAMES = figure-secretion.pdf figure-surface.pdf figure-serum.pdf figure-decontamination.pdf figure-decontamination-controls.pdf

FIGURE_PATHS = $(addprefix $(FIG_DIR)/, $(FIGURE_NAMES))

TABLE_NAMES = table-halflives-surface.tsv table-halflives-liquid.tsv table-halflives-surface.tex

TABLE_PATHS = $(addprefix $(TAB_DIR)/, $(TABLE_NAMES))


#####################################
# Rules
####################################

$(CLEANED_DIR)/%.tsv: $(SRC)/clean_data.py $(RAW_DIR)/%.xlsx
	$(MKDIR) $(CLEANED_DIR)
	./$^ $@

$(CHAIN_DIR)/%.pickle: $(SRC)/infer.py $(CLEANED_DIR)/mpx-data.tsv
	$(MKDIR) $(CHAIN_DIR)
	./$^ $@

$(FIG_DIR)/figure-decontamination-controls.pdf: $(SRC)/figure-decontamination.py $(CLEANED_DIR)/mpx-data.tsv $(DEFAULT_CHAIN_PATHS)
	$(MKDIR) $(FIG_DIR)
	./$^ $@

$(FIG_DIR)/%.png: $(SRC)/%.py $(CLEANED_DIR)/mpx-data.tsv $(DEFAULT_CHAIN_PATHS)
	$(MKDIR) $(FIG_DIR)
	./$^ $@
$(FIG_DIR)/%.pdf: $(SRC)/%.py $(CLEANED_DIR)/mpx-data.tsv $(DEFAULT_CHAIN_PATHS)
	$(MKDIR) $(FIG_DIR)
	./$^ $@

$(TAB_DIR)/table-halflives-%.tsv: $(SRC)/table-halflives.py $(CLEANED_DIR)/mpx-data.tsv $(DEFAULT_CHAIN_PATHS)
	$(MKDIR) $(TAB_DIR)
	./$^ $@

$(TAB_DIR)/table-halflives-%.tex: $(SRC)/table-halflives.py $(CLEANED_DIR)/mpx-data.tsv $(DEFAULT_CHAIN_PATHS)
	$(MKDIR) $(TAB_DIR)
	./$^ $@


.PHONY: install deltemp delfigs delchains clean chains figures tables all rebuild

## remove emacs tempfiles, etc.

# hidden files to remove
# for save directory delete
HIDDEN = .DS_Store

deltemp:
	$(RM) $(SRC)/*~*
	$(RM) $(SRC)/*#*
	$(RM) $(SRC)/$(HIDDEN)
	$(RM) -r $(SRC)/__pycache__

delfigs:
	$(MKDIR) $(FIG_DIR)
	$(RM) $(FIG_DIR)/*.pdf 
	$(RM) $(FIG_DIR)/*.png 
	$(RM) $(FIG_DIR)/$(HIDDEN)
	$(RMDIR) $(FIG_DIR)

delchains:
	$(MKDIR) $(CHAIN_DIR)
	$(RM) $(CHAIN_DIR)/*.pickle 
	$(RM) $(CHAIN_DIR)/$(HIDDEN)
	$(RMDIR) $(CHAIN_DIR)

deltabs:
	$(MKDIR) $(TAB_DIR)
	$(RM) $(TAB_DIR)/*.tsv
	$(RM) $(TAB_DIR)/*.tex
	$(RMDIR) $(TAB_DIR)

deldat:
	$(MKDIR) $(CLEANED_DIR)
	$(RM) $(CLEANED_DIR)/*.tsv
	$(RM) $(CLEANED_DIR)/$(HIDDEN)
	$(RMDIR) $(CLEANED_DIR)

clean: deltemp delfigs delchains deltabs deldat


.PHONY: install
install: requirements.txt
	$(PIP) install -r requirements.txt

data: $(CLEANED_DATA)
chains: $(ALL_CHAIN_PATHS)
figures: $(FIGURE_PATHS)
tables: $(TABLE_PATHS)

all: install chains figures tables
rebuild: clean all
