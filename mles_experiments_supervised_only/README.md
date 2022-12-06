Проводится эксперимент по анализу влияния объёма данных. Производится два вида претрейна: только на размеченных данных(supervised) и на всех данных (размеченные и неразмеченные) (supervised + unsup). Сравнение проводится на даунстрим задаче. В директории приведены два вида конфигов для датасета Gender. parquet_all обучается на всех данных, parquet_sup только на размеченных.

                           |     mean \pm std      |
    Gender auroc:
        supervised         |    0.879 \pm 0.005    |
        supervised + unsup |    0.884 \pm 0.006    |
        
    Gender accuracy:
        supervised         |    0.795 \pm 0.004    |
        supervised + unsup |    0.796 \pm 0.009    |
                         
    Age group (age_pred) accuracy:
        supervised         |    0.631 \pm 0.004    |
        supervised + unsup |    0.639 \pm 0.003    |
    
    Churn (rosbank) auroc:
        supervised         |    0.844 \pm 0.004    |
        supervised + unsup |    0.841 \pm 0.001    |
        
    Churn (rosbank) accuracy:
        supervised         |    0.764 \pm 0.009    |
        supervised + unsup |    0.762 \pm 0.010    |
    
    Retail (x5) accuracy:
        supervised         |    0.523 \pm 0.002    |
        supervised + unsup |           -           |
    
    Scoring (alpha battle) auroc:
        supervised         |    0.604 \pm 0.005    |
        supervised + unsup |    0.601 \pm 0.005    |


