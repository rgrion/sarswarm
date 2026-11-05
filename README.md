# SARSwarm — Busca cooperativa, detecção e sinalização de pessoas com microdrones

> Projeto de pesquisa do Colégio Militar do Rio de Janeiro (CMRJ), desenvolvido no contexto da FEBRACE 2027.
>
> **Estado atual:** mapeamento sistemático da literatura e experimento inicial de visão computacional concluídos. Busca cooperativa, comunicação entre agentes, formação em seta e integração física ainda não foram validadas.

## 1. Visão geral

O **SARSwarm** (*Search and Rescue Swarm*) investiga uma arquitetura de **microdrones de baixo custo** para apoiar a localização de pessoas em ambientes florestais. A proposta integra três funções:

1. **Busca cooperativa:** distribuir a exploração de uma região entre diferentes drones.
2. **Detecção visual:** identificar pessoas nas imagens obtidas durante a busca.
3. **Sinalização espacial:** após uma detecção, reorganizar os agentes em uma formação semelhante a uma **seta**, apontando para a posição identificada.

A pesquisa é conduzida em etapas. O primeiro componente avaliado experimentalmente foi o **detector de pessoas com YOLO11 Nano**, descrito neste documento. Os resultados apresentados **não** demonstram, por si só, o funcionamento de um enxame ou a eficácia de missões reais de busca e salvamento.

**Links do projeto:**

- Site institucional: https://sarswarm.s.gy
- Dashboard científico: https://sarswarm.s.gy/dashboard
- Repositório do projeto: https://sarswarm.s.gy/github

## 2. Fundamentação e planejamento

Um Mapeamento Sistemático da Literatura (MSL) orientou o desenvolvimento do SARSwarm. A busca realizada na base SciELO identificou **28 publicações**; após os critérios de seleção e avaliação metodológica, **11 estudos** foram incluídos. Foram investigados temas de coordenação e formação, comunicação, planejamento cooperativo, comportamentos autônomos e visão computacional.

Na amostra de 11 estudos, não foi identificada uma solução experimentalmente validada que integrasse simultaneamente busca cooperativa, detecção de alvos e sinalização por reconfiguração dinâmica em seta. Essa conclusão é **restrita à amostra e à estratégia de busca do MSL**, não a toda a literatura científica.

A hipótese central do SARSwarm é que a integração de algoritmos de busca, percepção visual e controle de formação poderá viabilizar uma arquitetura cooperativa acessível. A avaliação dessa hipótese depende de experimentos posteriores.

## 3. Experimento 02 — Detecção de pessoas em ambientes florestais

### 3.1 Objetivo

Treinar e avaliar um modelo leve de detecção de objetos para reconhecer **pessoas** em imagens de ambientes florestais, produzindo uma linha de base quantitativa para o futuro módulo de percepção do SARSwarm.

### 3.2 Conjunto de dados

Foi utilizado o conjunto **[`etri/ForestPersons`](https://huggingface.co/datasets/etri/ForestPersons)**, disponibilizado na plataforma Hugging Face. O acesso pode estar sujeito a autorização ou condições definidas pelos mantenedores do dataset.

A seleção foi reproduzível com **semente aleatória 42** e respeitou a separação dos subconjuntos originais, evitando utilizar as imagens de teste no treinamento ou na seleção dos pesos.

| Subconjunto | Imagens | Pessoas anotadas | Utilização |
|---|---:|---:|---|
| Treinamento | 500 | 1.047 | Ajuste dos parâmetros do modelo |
| Validação | 100 | 209 | Acompanhamento do treinamento e seleção de pesos |
| Teste independente | 100 | 194 | Avaliação final |
| **Total** | **700** | **1.450** | — |

As anotações de caixas delimitadoras foram convertidas do formato **COCO** para o formato **YOLO**, com coordenadas normalizadas e **uma única classe**, `person` (pessoa). O subconjunto preparado foi verificado quanto à integridade dos arquivos e das caixas delimitadoras.

> **Importante:** as imagens do ForestPersons não equivalem necessariamente a imagens capturadas por câmeras embarcadas em drones. A generalização para perspectivas aéreas permanece não avaliada.

### 3.3 Ambiente e configuração de treinamento

O treinamento foi realizado no **Google Colab**, com GPU **NVIDIA Tesla T4**, utilizando a biblioteca **Ultralytics 8.4.192** e pesos pré-treinados `yolo11n.pt`.

| Parâmetro | Valor |
|---|---|
| Modelo | YOLO11 Nano (`yolo11n`) |
| Inicialização | Pesos pré-treinados `yolo11n.pt` |
| Biblioteca | Ultralytics 8.4.192 |
| Épocas | 15 |
| Tamanho de entrada | 640 × 640 pixels |
| Batch size | 16 |
| Semente | 42 |
| Classe de interesse | `person` (1 classe) |
| Plataforma | Google Colab |
| GPU | NVIDIA Tesla T4 |

O modelo foi treinado nas **500 imagens de treinamento**, com acompanhamento nas **100 imagens de validação**. Ao término do treinamento, foi utilizado o arquivo `best.pt`, selecionado com base no desempenho de validação registrado pela rotina de treinamento.

**Fluxo resumido:**

```text
ForestPersons (Hugging Face)
          |
          v
Seleção reproduzível (seed = 42)
          |
          v
Conversão das anotações COCO -> YOLO
          |
          v
Treinamento (500 imagens) + validação (100 imagens)
          |
          v
Seleção dos pesos best.pt
          |
          v
Teste independente (100 imagens / 194 pessoas)
          |
          +--> Métricas globais
          +--> Tempos de processamento
          +--> Análise visual de 12 imagens
```

### 3.4 Avaliação e métricas

A avaliação final utilizou **100 imagens independentes**, não empregadas no treinamento nem na seleção dos pesos. Foram medidas:

- **Precisão (precision):** proporção de detecções positivas que correspondem a objetos reais.
- **Revocação (recall):** proporção de pessoas anotadas que o detector consegue recuperar.
- **mAP@50:** precisão média com limiar de interseção sobre união (IoU) de 0,50.
- **mAP@50–95:** média da precisão média em limiares de IoU de 0,50 a 0,95.

#### Resultados quantitativos

| Métrica | Validação | Teste independente |
|---|---:|---:|
| Precisão | 89,95% | **81,41%** |
| Recall | 77,99% | **81,24%** |
| mAP@50 | 86,51% | **84,31%** |
| mAP@50–95 | 51,81% | **50,07%** |

No teste independente, o modelo alcançou **84,31% de mAP@50**. O resultado inferior em **mAP@50–95 (50,07%)** indica que a localização das caixas torna-se menos precisa sob critérios de sobreposição mais rigorosos.

Em relação à validação, a precisão diminuiu **8,54 pontos percentuais**, enquanto o recall aumentou **3,25 pontos percentuais**. Essas diferenças descrevem a variação entre os subconjuntos, mas não determinam, isoladamente, sua causa.

#### Tempo médio de processamento

| Etapa | Tempo por imagem |
|---|---:|
| Pré-processamento | 6,32 ms |
| Inferência | 11,39 ms |
| Pós-processamento | 6,05 ms |
| **Total** | **23,76 ms** |

Esses tempos foram medidos na **Tesla T4**. Não devem ser interpretados como velocidade de inferência em ESP32, microcontroladores, computadores embarcados ou microdrones físicos. Também não incluem necessariamente toda a latência de um sistema aéreo integrado (captura, transmissão, decisão e atuação).

### 3.5 Análise visual complementar — 12 imagens

Foi realizada uma inspeção de **12 imagens** pertencentes ao conjunto independente de teste. As detecções foram comparadas às anotações com:

- Confiança mínima: **0,25**;
- IoU mínimo para correspondência: **0,50**;
- Associação de cada previsão a, no máximo, uma anotação de referência.

| Indicador | Quantidade |
|---|---:|
| Pessoas anotadas (GT) | 19 |
| Previsões produzidas | 19 |
| Verdadeiros positivos (TP) | 17 |
| Falsos positivos (FP) | 2 |
| Falsos negativos (FN) | 2 |
| Precisão nesta amostra | 89,47% |
| Recall nesta amostra | 89,47% |

As métricas de **89,47%** correspondem **somente a essas 12 imagens ilustrativas**; não substituem os valores de precisão e recall calculados sobre o teste completo de 100 imagens.

**Galerias produzidas no experimento:**

1. [Galeria de previsões do YOLO11 Nano](https://drive.google.com/file/d/1m-ynZULafQw9hF9fDyWQ_H42xbEzT418/view)
2. [Galeria científica de comparação — TP, FP e FN](https://drive.google.com/file/d/1k0St5kJi9ostIlpY4W0gNtS0ffXl2LxQ/view)

Na galeria científica, foram usadas as seguintes cores de identificação: **verde** para verdadeiros positivos, **laranja** para falsos positivos e **vermelho** para falsos negativos.

## 4. O que os resultados demonstram — e o que ainda não demonstram

### Evidências obtidas

- Foi possível preparar um subconjunto anotado e reproduzível de **700 imagens** para detecção de pessoas.
- O **YOLO11 Nano** foi treinado durante **15 épocas** e avaliado com um conjunto independente.
- O detector apresentou **81,41% de precisão**, **81,24% de recall** e **84,31% de mAP@50** no teste independente.
- Foram documentados exemplos de acertos e erros, incluindo falsos negativos relevantes para a aplicação pretendida.

### Limitações

- **Não foi avaliada a operação do modelo em um microdrone físico.**
- **Não foi medida a inferência em hardware embarcado.**
- **Não foi validada a detecção com imagens aéreas representativas da missão.**
- **Não foram demonstradas a busca cooperativa, a comunicação multiagente ou a formação em seta.**
- Os resultados de um detector de objetos **não comprovam**, isoladamente, a viabilidade operacional de uma solução de busca e salvamento.

Os **falsos negativos** merecem atenção especial, pois representam pessoas presentes nas imagens que não foram detectadas. Etapas futuras deverão investigar diferentes cenários, perspectivas e condições de iluminação antes de qualquer conclusão sobre uso operacional.

## 5. Integração prevista ao SARSwarm

O detector constitui o módulo de **percepção visual** da arquitetura proposta. A integração futura deverá seguir uma lógica semelhante a esta:

```text
[Microdrones em busca cooperativa]
                 |
                 v
        [Captura de imagens]
                 |
                 v
     [Detecção visual de pessoas]
                 |
         Alvo identificado?
            /          \
          Não          Sim
           |            |
           v            v
  [Continuar busca] [Estimar posição do alvo]
                        |
                        v
               [Compartilhar informação]
                        |
                        v
             [Reorganizar os agentes]
                        |
                        v
             [Formação de seta / sinalização]
```

Esse fluxo é **conceitual** e não corresponde a um sistema integrado já implementado ou testado. A transformação de detecções em coordenadas espaciais confiáveis, a comunicação entre os drones e a segurança da formação exigirão investigação específica.

### Próximas etapas planejadas

1. Simular a exploração cooperativa com **1, 3, 5, 7 e 10 drones**.
2. Comparar cobertura espacial, tempo de busca e sobreposição de trajetórias.
3. Investigar comunicação e coordenação entre agentes, inclusive diante de falhas.
4. Implementar e avaliar a transição para uma **formação geométrica em seta**.
5. Testar a generalização do detector para imagens aéreas.
6. Avaliar processamento, autonomia, custo e segurança de possíveis protótipos físicos.

## 6. Reprodutibilidade e materiais necessários

Para reproduzir o Experimento 02 com fidelidade, são necessários:

- Acesso autorizado ao dataset [`etri/ForestPersons`](https://huggingface.co/datasets/etri/ForestPersons), quando aplicável;
- Código de seleção e conversão das anotações de COCO para YOLO;
- Identificação exata das imagens selecionadas para os três subconjuntos;
- Configuração de treinamento (tabela da seção 3.3);
- Versões das dependências e ambiente computacional;
- Pesos finais `best.pt` e registros de treinamento, caso disponibilizados;
- Scripts e saídas de avaliação do teste independente;
- Galerias de análise visual e seus critérios de correspondência.

**Transparência:** este README documenta os parâmetros e resultados conhecidos do experimento. A presença efetiva de notebooks, scripts, pesos, arquivos de configuração e dados no repositório deve ser conferida separadamente. Não se afirma aqui que esses arquivos estejam publicados. Os dados originais devem ser compartilhados apenas conforme a licença e as condições de acesso do conjunto ForestPersons.

## 7. Referências e documentação

- **ETRI.** *ForestPersons*. Dataset disponível no Hugging Face. https://huggingface.co/datasets/etri/ForestPersons
- **ULTRALYTICS.** *Ultralytics YOLO11*. Documentação técnica. https://docs.ultralytics.com/models/yolo11/
- **REDMON, J.; FARHADI, A.** YOLOv3: An Incremental Improvement. *arXiv*, arXiv:1804.02767, 2018. https://arxiv.org/abs/1804.02767
- **BRAMBILLA, M.; FERRANTE, E.; BIRATTARI, M.; DORIGO, M.** Swarm robotics: a review from the swarm engineering perspective. *Swarm Intelligence*, v. 7, p. 1–41, 2013. https://doi.org/10.1007/s11721-012-0075-2
- **HAGHIGHI, H. et al.** Hybrid form of particle swarm optimization and genetic algorithm for optimal path planning in coverage mission by cooperated unmanned aerial vehicles. *Journal of Aerospace Technology and Management*, v. 12, e4320, 2020. https://doi.org/10.5028/jatm.v12.1169
- **XUE, D.; CAI, G.** Formation flight control of multi-UAV system with communication constraints. *Journal of Aerospace Technology and Management*, v. 8, n. 2, 2016. https://doi.org/10.5028/jatm.v8i2.608
- **KITCHENHAM, B.; CHARTERS, S.** *Guidelines for performing systematic literature reviews in software engineering*. Technical Report EBSE-2007-01, 2007.
- **PAGE, M. J. et al.** The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, v. 372, n71, 2021. https://doi.org/10.1136/bmj.n71

## 8. Equipe

**Colégio Militar do Rio de Janeiro (CMRJ)**

**Alunos-pesquisadores:** Rian Costa Vecchioni, Rafael Rocha Leira e Miguel Marques Duarte Alves.

**Orientador:** Alexandre dos Santos Boente.

**Coorientador:** Renan Figueira Grion.

---

*SARSwarm — pesquisa em desenvolvimento. Este documento descreve resultados experimentais iniciais e uma arquitetura proposta; não representa um sistema de resgate operacional validado.*
