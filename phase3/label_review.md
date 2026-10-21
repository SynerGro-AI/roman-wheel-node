# SciFact label review (for a human; nothing here changes any label or any reported number)

Claims where **every** Grok judge (8-9 judges, 5 models, 6 prompt styles) reading the *cited* abstracts gave the
same verdict, and that verdict differs from the SciFact gold label. Unanimity is not proof: the panel is
correlated (error φ 0.4-0.9), so a shared misreading is possible. Each case needs a domain expert.

Dev (held-out): 9 of 300 claims (3.0%). Train subset: 8 of 350 (2.3%).

## dev

### dev:1385: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.90)
- Claim: cSMAC formation enhances weak ligand signalling.
- Cited abstract 306006: *The stimulatory potency of T cell antigens is influenced by the formation of the immunological synapse.*
- Judge quote (verbatim from the abstract): "This conclusion was supported by experiments that showed that enhancing cSMAC formation reduced stimulatory capacity of the weak peptide."
- Gold rationale sentence: "This conclusion was supported by experiments that showed that enhancing cSMAC formation reduced stimulatory capacity of the weak peptide."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:593: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.90)
- Claim: Incidence of heart failure decreased by 10% in women since 1979.
- Cited abstract 19675911: *Trends in heart failure incidence and survival in a community-based population.*
- Judge quote (verbatim from the abstract): "The incidence of heart failure was higher among men (378/100 000 persons; 95% confidence interval [CI], 361-395 for men; 289/100 000 persons; 95% CI, 277-300 for women) and did not change over time among men or women."
- Gold rationale sentence: "RESULTS The incidence of heart failure was higher among men (378/100 000 persons; 95% confidence interval [CI], 361-395 for men; 289/100 000 persons; 95% CI, 277-300 for women) and did not change over time among men or women."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:808: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.85)
- Claim: Most termination events in Okazaki fragments are sequence specific.
- Cited abstract 36606083: *Quantitative, genome-wide analysis of eukaryotic replication initiation and termination.*
- Judge quote (verbatim from the abstract): "Using a strain in which late origins can be induced to fire early, we show that replication termination is a largely passive phenomenon that does not rely on cis-acting sequences or replication fork pausing."
- Gold rationale sentence: "Via the deep sequencing of Okazaki fragments, we comprehensively document replication fork directionality throughout the S. cerevisiae genome, which permits the systematic analysis of initiation, origin efficiency, fork progression, and termination."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:907: gold **NEI**, panel **SUPPORT** (min judge confidence 0.85)
- Claim: PGE 2 promotes intestinal tumor growth by altering the expression of tumor suppressing and DNA repair genes.
- Cited abstract 6923961: *Prostaglandin E2 promotes intestinal tumor growth via DNA methylation*
- Judge quote (verbatim from the abstract): "Here we show that prostaglandin E(2) (PGE(2)) silences certain tumor-suppressor and DNA-repair genes through DNA methylation to promote tumor growth."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:1368: gold **SUPPORT**, panel **NEI** (min judge confidence 0.80)
- Claim: Vitamin D deficiency effects the term of delivery.
- Cited abstract 2425364: *Association between maternal serum 25-hydroxyvitamin D level and pregnancy and neonatal outcomes: systematic review and meta-analysis of observational studies.*
- Gold rationale sentence: "Insufficient serum levels of 25-OHD were associated with gestational diabetes (pooled odds ratio 1.49, 95% confidence interval 1.18 to 1.89), pre-eclampsia (1.79, 1.25 to 2.58), and small for gestational age infants (1.85, 1.52 to 2.26)."
- Gold rationale sentence: "Pregnant women with low serum 25-OHD levels had an increased risk of bacterial vaginosis and low birthweight infants but not delivery by caesarean section.   
"
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:783: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.80)
- Claim: Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA.
- Cited abstract 40632104: *Dual role of the IL-12/IFN-gamma axis in the development of autoimmune myocarditis: induction by IL-12 and protection by IFN-gamma.*
- Judge quote (verbatim from the abstract): "However, IFN-gamma-deficient mice developed fatal autoimmune disease"
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:879: gold **CONTRADICT**, panel **SUPPORT** (min judge confidence 0.75)
- Claim: Occupancy of ribosomes by IncRNAs do not make functional peptides.
- Cited abstract 8426046: *Ribosome Profiling Provides Evidence that Large Noncoding RNAs Do Not Encode Proteins*
- Judge quote (verbatim from the abstract): "Taken together, these results argue that the large majority of lincRNAs do not function through encoded proteins."
- Gold rationale sentence: "Here, we show that classical noncoding RNAs and 5' UTRs show the same ribosome occupancy as lincRNAs, demonstrating that ribosome occupancy alone is not sufficient to classify transcripts as coding or noncoding."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:1292: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.70)
- Claim: There is no association between HNF4A mutations and diabetes risks.
- Cited abstract 56893404: *Macrosomia and Hyperinsulinaemic Hypoglycaemia in Patients with Heterozygous Mutations in the HNF4A Gene*
- Judge quote (verbatim from the abstract): "We assessed the in utero and neonatal role of two key regulators of pancreatic insulin secretion by studying birthweight and the incidence of neonatal hypoglycaemia in patients with heterozygous mutations in the maturity-onset diabetes of the young (MODY) genes HNF4A (encoding HNF-4α) and HNF1A/TCF1 (encoding HNF-1α), and the effect of pancreatic deletion of Hnf4a on foetal and neonatal insulin secretion in mice."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:693: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.55)
- Claim: Leuko-reduced blood reduces infectious complications in red blood cell transfusion.
- Cited abstract 24088502: *Clinical outcomes following institution of the Canadian universal leukoreduction program for red blood cell transfusions.*
- Judge quote (verbatim from the abstract): "but serious nosocomial infections did not decrease (adjusted OR, 0.97; 95% confidence interval [CI], 0.87-1.09)."
- Gold rationale sentence: "Compared with the control period, the adjusted odds of death following leukoreduction were reduced (odds ratio [OR], 0.87; 95% confidence interval [CI], 0.75-0.99), but serious nosocomial infections did not decrease (adjusted OR, 0.97; 95% CI, 0.87-1.09)."
- Gold rationale sentence: "The frequency of posttransfusion fevers decreased significantly following leukoreduction (adjusted OR, 0.86; 95% CI, 0.79-0.94), as did antibiotic use (adjusted OR, 0.90; 95% CI, 0.82-0.99).   
"
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

## train

### train:44: gold **SUPPORT**, panel **NEI** (min judge confidence 0.85)
- Claim: A mutation in HNF4A leads to an increased risk of developing diabetes by the age of 14 years.
- Cited abstract 56893404: *Macrosomia and Hyperinsulinaemic Hypoglycaemia in Patients with Heterozygous Mutations in the HNF4A Gene*
- Gold rationale sentence: "We assessed the in utero and neonatal role of two key regulators of pancreatic insulin secretion by studying birthweight and the incidence of neonatal hypoglycaemia in patients with heterozygous mutations in the maturity-onset diabetes of the young (MODY) genes HNF4A (encoding HNF-4α) and HNF1A/TCF1 (encoding HNF-1α), and the effect of pancreatic deletion of Hnf4a on foetal and neonatal insulin secretion in mice."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:1386: gold **CONTRADICT**, panel **SUPPORT** (min judge confidence 0.82)
- Claim: cSMAC formation represses weak ligand signalling.
- Cited abstract 306006: *The stimulatory potency of T cell antigens is influenced by the formation of the immunological synapse.*
- Judge quote (verbatim from the abstract): "This conclusion was supported by experiments that showed that enhancing cSMAC formation reduced stimulatory capacity of the weak peptide."
- Gold rationale sentence: "This conclusion was supported by experiments that showed that enhancing cSMAC formation reduced stimulatory capacity of the weak peptide."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:1214: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.80)
- Claim: The deregulated and prolonged activation of monocytes improves inflammatory disease outcomes.
- Cited abstract 6493422: *The myeloid transcription factor KLF2 regulates the host response to polymicrobial infection and endotoxic shock.*
- Judge quote (verbatim from the abstract): "this activation process must be under exquisite control to prevent uncontrolled inflammation."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:989: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.77)
- Claim: Pure neural progenitor cell (NPC) populations can only be obtained from cell cultures that undergo passaging, filtration, or other isolation and cell sorting methods.
- Cited abstract 9988425: *Niche-Independent Symmetrical Self-Renewal of a Mammalian Tissue Stem Cell*
- Judge quote (verbatim from the abstract): "Here we demonstrate that the complex neurosphere environment is dispensable for stem cell maintenance, and that the combination of fibroblast growth factor 2 (FGF-2) and epidermal growth factor (EGF) is sufficient for derivation and continuous expansion by symmetrical division of pure cultures of neural stem (NS) cells."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:651: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.75)
- Claim: Interferon stimulated genes (ISGs) Irf1, Irg1, Ifi27, and Rsad2 encourage West Nile virus replication in cortical neurons.
- Cited abstract 9433958: *Differential innate immune response programs in neuronal subtypes determine susceptibility to infection in the brain by positive stranded RNA viruses*
- Judge quote (verbatim from the abstract): "By transducing cortical neurons with genes that were expressed more highly in granule cell neurons, we identified three interferon-stimulated genes (ISGs; Ifi27, Irg1 and Rsad2 (also known as Viperin)) that mediated the antiviral effects against different neurotropic viruses."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:1193: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.75)
- Claim: The appearance of brown-like or beige cells primarily occurs in visceral fat, not subcutaneous fat.
- Cited abstract 20532591: *Tracking adipogenesis during white adipose tissue development, expansion and regeneration*
- Judge quote (verbatim from the abstract): "During cold-induced 'browning' of subcutaneous fat, most 'beige' adipocytes stem from de novo–differentiated adipocytes."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:169: gold **CONTRADICT**, panel **NEI** (min judge confidence 0.70)
- Claim: Bariatric surgery reduces postmenopausal breast cancer.
- Cited abstract 5824985: *Bariatric Surgery in the United Kingdom: A Cohort Study of Weight Loss and Clinical Outcomes in Routine Clinical Care.*
- Judge quote (verbatim from the abstract): "No association was detected between bariatric surgery and fractures, cancer, or stroke."
- Gold rationale sentence: "No association was detected between bariatric surgery and fractures, cancer, or stroke."
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:610: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.63)
- Claim: Increased flux of microbial products suppresses immune responses.
- Cited abstract 40096222: *Compromised intestinal epithelial barrier induces adaptive immune compensation that protects from colitis.*
- Judge quote (verbatim from the abstract): "Mice lacking junctional adhesion molecule A (JAM-A, encoded by F11r) exhibit enhanced intestinal epithelial permeability, bacterial translocation, and elevated colonic lymphocyte numbers, yet do not develop colitis."
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

