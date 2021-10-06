
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d600b630",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error,mean_squared_error\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.dummy import DummyRegressor\n",
    "import warnings\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b617b895",
   "metadata": {},
   "outputs": [],
   "source": [
    "warnings.simplefilter('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a5a4bca4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: adjdatatools in c:\\users\\alexander\\anaconda3\\lib\\site-packages (0.4.0)\n",
      "Requirement already satisfied: numpy in c:\\users\\alexander\\anaconda3\\lib\\site-packages (from adjdatatools) (1.18.5)\n",
      "Requirement already satisfied: pandas>=0.20.0 in c:\\users\\alexander\\anaconda3\\lib\\site-packages (from adjdatatools) (1.2.4)\n",
      "Requirement already satisfied: pytz>=2017.3 in c:\\users\\alexander\\anaconda3\\lib\\site-packages (from pandas>=0.20.0->adjdatatools) (2021.1)\n",
      "Requirement already satisfied: python-dateutil>=2.7.3 in c:\\users\\alexander\\anaconda3\\lib\\site-packages (from pandas>=0.20.0->adjdatatools) (2.8.1)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\alexander\\anaconda3\\lib\\site-packages (from python-dateutil>=2.7.3->pandas>=0.20.0->adjdatatools) (1.15.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install adjdatatools"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8cc18a1c",
   "metadata": {},
   "source": [
    "### *Загрузка данных за 2011 - 2015 г.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "41acf437",
   "metadata": {},
   "outputs": [],
   "source": [
    "gas_2011 = pd.read_csv('Датасеты/gt_2011.txt')\n",
    "gas_2012 = pd.read_csv('Датасеты/gt_2012.txt')\n",
    "gas_2013 = pd.read_csv('Датасеты/gt_2013.txt')\n",
    "gas_2014 = pd.read_csv('Датасеты/gt_2014.txt')\n",
    "gas_2015 = pd.read_csv('Датасеты/gt_2015.txt')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea2411de",
   "metadata": {},
   "source": [
    "### *Расшифровка обозначений*:\n",
    "            Ambient temperature (AT) C Температура окружающей среды\n",
    "            Ambient pressure (AP) mbar Давление окружающей среды\n",
    "            Ambient humidity (AH) (%) Влажность окружающего воздуха\n",
    "            Air filter difference pressure (AFDP) mbar Перепад давления в воздушном фильтре\n",
    "            Gas turbine exhaust pressure (GTEP) mbar Конечное давление\n",
    "            Turbine inlet temperature (TIT) C Температура газов на входе в турбину\n",
    "            Turbine after temperature (TAT) C Температура газов на выходе из турбины\n",
    "            Compressor discharge pressure (CDP) mbar Конечное давление компрессора\n",
    "            Turbine energy yield (TEY) MWH Выработка электроэнергии\n",
    "            Carbon monoxide (CO) mg/m3 Концентрация окиси углерода \n",
    "            Nitrogen oxides (NOx) mg/m3 Концентрация окиси азота"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "56e1eca6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AT</th>\n",
       "      <th>AP</th>\n",
       "      <th>AH</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>TEY</th>\n",
       "      <th>CDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>NOX</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "      <td>7411.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>17.112261</td>\n",
       "      <td>1014.167883</td>\n",
       "      <td>79.174989</td>\n",
       "      <td>4.090755</td>\n",
       "      <td>25.663721</td>\n",
       "      <td>1084.733909</td>\n",
       "      <td>544.503170</td>\n",
       "      <td>135.745675</td>\n",
       "      <td>12.207578</td>\n",
       "      <td>1.572486</td>\n",
       "      <td>67.575392</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7.428307</td>\n",
       "      <td>6.293003</td>\n",
       "      <td>13.465898</td>\n",
       "      <td>0.661865</td>\n",
       "      <td>4.325835</td>\n",
       "      <td>16.134972</td>\n",
       "      <td>8.288471</td>\n",
       "      <td>16.209187</td>\n",
       "      <td>1.146561</td>\n",
       "      <td>1.845442</td>\n",
       "      <td>10.683331</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.116300</td>\n",
       "      <td>995.790000</td>\n",
       "      <td>34.493000</td>\n",
       "      <td>2.758400</td>\n",
       "      <td>17.878000</td>\n",
       "      <td>1000.800000</td>\n",
       "      <td>512.450000</td>\n",
       "      <td>100.170000</td>\n",
       "      <td>9.904400</td>\n",
       "      <td>0.000388</td>\n",
       "      <td>27.765000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>10.992000</td>\n",
       "      <td>1009.800000</td>\n",
       "      <td>70.428500</td>\n",
       "      <td>3.644750</td>\n",
       "      <td>23.364500</td>\n",
       "      <td>1082.500000</td>\n",
       "      <td>538.560000</td>\n",
       "      <td>130.745000</td>\n",
       "      <td>11.684000</td>\n",
       "      <td>0.726405</td>\n",
       "      <td>60.361500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>16.366000</td>\n",
       "      <td>1013.600000</td>\n",
       "      <td>82.129000</td>\n",
       "      <td>4.026300</td>\n",
       "      <td>24.770000</td>\n",
       "      <td>1088.000000</td>\n",
       "      <td>549.860000</td>\n",
       "      <td>133.810000</td>\n",
       "      <td>12.008000</td>\n",
       "      <td>1.155700</td>\n",
       "      <td>65.542000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>23.344500</td>\n",
       "      <td>1018.100000</td>\n",
       "      <td>89.778000</td>\n",
       "      <td>4.480350</td>\n",
       "      <td>29.879500</td>\n",
       "      <td>1099.800000</td>\n",
       "      <td>550.040000</td>\n",
       "      <td>148.325000</td>\n",
       "      <td>13.257000</td>\n",
       "      <td>1.754600</td>\n",
       "      <td>74.314500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>34.532000</td>\n",
       "      <td>1034.200000</td>\n",
       "      <td>100.170000</td>\n",
       "      <td>7.318900</td>\n",
       "      <td>36.003000</td>\n",
       "      <td>1100.600000</td>\n",
       "      <td>550.610000</td>\n",
       "      <td>170.000000</td>\n",
       "      <td>14.851000</td>\n",
       "      <td>43.622000</td>\n",
       "      <td>119.320000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                AT           AP           AH         AFDP         GTEP  \\\n",
       "count  7411.000000  7411.000000  7411.000000  7411.000000  7411.000000   \n",
       "mean     17.112261  1014.167883    79.174989     4.090755    25.663721   \n",
       "std       7.428307     6.293003    13.465898     0.661865     4.325835   \n",
       "min       2.116300   995.790000    34.493000     2.758400    17.878000   \n",
       "25%      10.992000  1009.800000    70.428500     3.644750    23.364500   \n",
       "50%      16.366000  1013.600000    82.129000     4.026300    24.770000   \n",
       "75%      23.344500  1018.100000    89.778000     4.480350    29.879500   \n",
       "max      34.532000  1034.200000   100.170000     7.318900    36.003000   \n",
       "\n",
       "               TIT          TAT          TEY          CDP           CO  \\\n",
       "count  7411.000000  7411.000000  7411.000000  7411.000000  7411.000000   \n",
       "mean   1084.733909   544.503170   135.745675    12.207578     1.572486   \n",
       "std      16.134972     8.288471    16.209187     1.146561     1.845442   \n",
       "min    1000.800000   512.450000   100.170000     9.904400     0.000388   \n",
       "25%    1082.500000   538.560000   130.745000    11.684000     0.726405   \n",
       "50%    1088.000000   549.860000   133.810000    12.008000     1.155700   \n",
       "75%    1099.800000   550.040000   148.325000    13.257000     1.754600   \n",
       "max    1100.600000   550.610000   170.000000    14.851000    43.622000   \n",
       "\n",
       "               NOX  \n",
       "count  7411.000000  \n",
       "mean     67.575392  \n",
       "std      10.683331  \n",
       "min      27.765000  \n",
       "25%      60.361500  \n",
       "50%      65.542000  \n",
       "75%      74.314500  \n",
       "max     119.320000  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gas_2011.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c3468a15",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AT</th>\n",
       "      <th>AP</th>\n",
       "      <th>AH</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>TEY</th>\n",
       "      <th>CDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>NOX</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "      <td>7628.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>18.397950</td>\n",
       "      <td>1012.258153</td>\n",
       "      <td>79.074804</td>\n",
       "      <td>4.306717</td>\n",
       "      <td>25.181361</td>\n",
       "      <td>1082.890233</td>\n",
       "      <td>546.263793</td>\n",
       "      <td>132.675552</td>\n",
       "      <td>12.000121</td>\n",
       "      <td>2.361133</td>\n",
       "      <td>68.788965</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7.661038</td>\n",
       "      <td>6.384291</td>\n",
       "      <td>14.105087</td>\n",
       "      <td>0.831170</td>\n",
       "      <td>4.006825</td>\n",
       "      <td>16.852148</td>\n",
       "      <td>7.331345</td>\n",
       "      <td>15.302140</td>\n",
       "      <td>1.049384</td>\n",
       "      <td>2.474116</td>\n",
       "      <td>10.224937</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.522300</td>\n",
       "      <td>985.850000</td>\n",
       "      <td>30.344000</td>\n",
       "      <td>2.087400</td>\n",
       "      <td>18.100000</td>\n",
       "      <td>1024.600000</td>\n",
       "      <td>513.060000</td>\n",
       "      <td>101.150000</td>\n",
       "      <td>9.928600</td>\n",
       "      <td>0.007505</td>\n",
       "      <td>41.777000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>12.269250</td>\n",
       "      <td>1008.400000</td>\n",
       "      <td>69.165250</td>\n",
       "      <td>3.882650</td>\n",
       "      <td>23.090750</td>\n",
       "      <td>1075.800000</td>\n",
       "      <td>547.657500</td>\n",
       "      <td>125.240000</td>\n",
       "      <td>11.469750</td>\n",
       "      <td>1.127125</td>\n",
       "      <td>62.531250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>19.120500</td>\n",
       "      <td>1012.000000</td>\n",
       "      <td>82.411500</td>\n",
       "      <td>4.298050</td>\n",
       "      <td>25.221000</td>\n",
       "      <td>1089.100000</td>\n",
       "      <td>549.920000</td>\n",
       "      <td>133.760000</td>\n",
       "      <td>12.041000</td>\n",
       "      <td>1.636300</td>\n",
       "      <td>67.246500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>24.337250</td>\n",
       "      <td>1016.000000</td>\n",
       "      <td>90.356250</td>\n",
       "      <td>4.670525</td>\n",
       "      <td>26.322250</td>\n",
       "      <td>1093.800000</td>\n",
       "      <td>550.070000</td>\n",
       "      <td>134.900000</td>\n",
       "      <td>12.290000</td>\n",
       "      <td>2.953025</td>\n",
       "      <td>73.424500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>34.929000</td>\n",
       "      <td>1031.800000</td>\n",
       "      <td>100.200000</td>\n",
       "      <td>7.610600</td>\n",
       "      <td>37.402000</td>\n",
       "      <td>1100.800000</td>\n",
       "      <td>550.530000</td>\n",
       "      <td>174.610000</td>\n",
       "      <td>15.081000</td>\n",
       "      <td>44.103000</td>\n",
       "      <td>119.890000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                AT           AP           AH         AFDP         GTEP  \\\n",
       "count  7628.000000  7628.000000  7628.000000  7628.000000  7628.000000   \n",
       "mean     18.397950  1012.258153    79.074804     4.306717    25.181361   \n",
       "std       7.661038     6.384291    14.105087     0.831170     4.006825   \n",
       "min       0.522300   985.850000    30.344000     2.087400    18.100000   \n",
       "25%      12.269250  1008.400000    69.165250     3.882650    23.090750   \n",
       "50%      19.120500  1012.000000    82.411500     4.298050    25.221000   \n",
       "75%      24.337250  1016.000000    90.356250     4.670525    26.322250   \n",
       "max      34.929000  1031.800000   100.200000     7.610600    37.402000   \n",
       "\n",
       "               TIT          TAT          TEY          CDP           CO  \\\n",
       "count  7628.000000  7628.000000  7628.000000  7628.000000  7628.000000   \n",
       "mean   1082.890233   546.263793   132.675552    12.000121     2.361133   \n",
       "std      16.852148     7.331345    15.302140     1.049384     2.474116   \n",
       "min    1024.600000   513.060000   101.150000     9.928600     0.007505   \n",
       "25%    1075.800000   547.657500   125.240000    11.469750     1.127125   \n",
       "50%    1089.100000   549.920000   133.760000    12.041000     1.636300   \n",
       "75%    1093.800000   550.070000   134.900000    12.290000     2.953025   \n",
       "max    1100.800000   550.530000   174.610000    15.081000    44.103000   \n",
       "\n",
       "               NOX  \n",
       "count  7628.000000  \n",
       "mean     68.788965  \n",
       "std      10.224937  \n",
       "min      41.777000  \n",
       "25%      62.531250  \n",
       "50%      67.246500  \n",
       "75%      73.424500  \n",
       "max     119.890000  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gas_2012.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "63273c90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AT</th>\n",
       "      <th>AP</th>\n",
       "      <th>AH</th>\n",
       "      <th>AFDP</th>\n",
       "      <th>GTEP</th>\n",
       "      <th>TIT</th>\n",
       "      <th>TAT</th>\n",
       "      <th>TEY</th>\n",
       "      <th>CDP</th>\n",
       "      <th>CO</th>\n",
       "      <th>NOX</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>7152.00000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "      <td>7152.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>17.60262</td>\n",
       "      <td>1011.999607</td>\n",
       "      <td>80.461624</td>\n",
       "      <td>3.695958</td>\n",
       "      <td>25.105097</td>\n",
       "      <td>1081.569463</td>\n",
       "      <td>545.780885</td>\n",
       "      <td>132.168342</td>\n",
       "      <td>11.971586</td>\n",
       "      <td>2.723031</td>\n",
       "      <td>70.007899</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>6.86289</td>\n",
       "      <td>6.290065</td>\n",
       "      <td>14.125390</td>\n",
       "      <td>0.805829</td>\n",
       "      <td>4.350711</td>\n",
       "      <td>17.385147</td>\n",
       "      <td>7.358935</td>\n",
       "      <td>16.348156</td>\n",
       "      <td>1.132159</td>\n",
       "      <td>2.363913</td>\n",
       "      <td>12.048249</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.28985</td>\n",
       "      <td>989.380000</td>\n",
       "      <td>27.504000</td>\n",
       "      <td>2.329500</td>\n",
       "      <td>18.104000</td>\n",
       "      <td>1022.100000</td>\n",
       "      <td>518.320000</td>\n",
       "      <td>101.480000</td>\n",
       "      <td>9.875400</td>\n",
       "      <td>0.005033</td>\n",
       "      <td>43.198000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>12.04875</td>\n",
       "      <td>1008.400000</td>\n",
       "      <td>71.493500</td>\n",
       "      <td>3.100350</td>\n",
       "      <td>21.385000</td>\n",
       "      <td>1065.975000</td>\n",
       "      <td>543.745000</td>\n",
       "      <td>118.005000</td>\n",
       "      <td>11.001250</td>\n",
       "      <td>1.257975</td>\n",
       "      <td>62.269000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>17.20450</td>\n",
       "      <td>1011.800000</td>\n",
       "      <td>84.002000</td>\n",
       "      <td>3.627850</td>\n",
       "      <td>24.852500</td>\n",
       "      <td>1087.300000</td>\n",
       "      <td>549.900000</td>\n",
       "      <td>133.570000</td>\n",
       "      <td>11.956000</td>\n",
       "      <td>1.782700</td>\n",
       "      <td>68.651000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>23.16400</td>\n",
       "      <td>1016.000000</td>\n",
       "      <td>91.579000</td>\n",
       "      <td>4.156825</td>\n",
       "      <td>26.385750</td>\n",
       "      <td>1094.400000</td>\n",
       "      <td>550.030000</td>\n",
       "      <td>135.520000</td>\n",
       "      <td>12.319250</td>\n",
       "      <td>3.591225</td>\n",
       "      <td>76.001500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>33.87300</td>\n",
       "      <td>1029.700000</td>\n",
       "      <td>100.190000</td>\n",
       "      <td>6.977900</td>\n",
       "      <td>36.950000</td>\n",
       "      <td>1100.500000</td>\n",
       "      <td>550.530000</td>\n",
       "      <td>172.960000</td>\n",
       "      <td>14.867000</td>\n",
       "      <td>35.045000</td>\n",
       "      <td>119.910000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               AT           AP           AH         AFDP         GTEP  \\\n",
       "count  7152.00000  7152.000000  7152.000000  7152.000000  7152.000000   \n",
       "mean     17.60262  1011.999607    80.461624     3.695958    25.105097   \n",
       "std       6.86289     6.290065    14.125390     0.805829     4.350711   \n",
       "min       0.28985   989.380000    27.504000     2.329500    18.104000   \n",
       "25%      12.04875  1008.400000    71.493500     3.100350    21.385000   \n",
       "50%      17.20450  1011.800000    84.002000     3.627850    24.852500   \n",
       "75%      23.16400  1016.000000    91.579000     4.156825    26.385750   \n",
       "max      33.87300  1029.700000   100.190000     6.977900    36.950000   \n",
       "\n",
       "               TIT          TAT          TEY          CDP           CO  \\\n",
       "count  7152.000000  7152.000000  7152.000000  7152.000000  7152.000000   \n",
       "mean   1081.569463   545.780885   132.168342    11.971586     2.723031   \n",
       "std      17.385147     7.358935    16.348156     1.132159     2.363913   \n",
       "min    1022.100000   518.320000   101.480000     9.875400     0.005033   \n",
       "25%    1065.975000   543.745000   118.005000    11.001250     1.257975   \n",
       "50%    1087.300000   549.900000   133.570000    11.956000     1.782700   \n",
       "75%    1094.400000   550.030000   135.520000    12.319250     3.591225   \n",
       "max    1100.500000   550.530000   172.960000    14.867000    35.045000   \n",
       "\n",
       "               NOX  \n",
       "count  7152.000000  \n",
       "mean     70.007899  \n",
       "std      12.048249  \n",
       "min      43.198000  \n",
       "25%      62.269000  \n",
       "50%      68.651000  \n",
       "75%      76.001500  \n",
       "max     119.910000  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gas_2013.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d5fe04f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "gas_train = pd.DataFrame()\n",
    "gas_2011['Year'] = 2011\n",
    "gas_2012['Year'] = 2012\n",
    "gas_2013['Year'] = 2013\n",
    "gas_train = (pd.concat([gas_2011,gas_2012,gas_2013], ignore_index=True))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eeebdd63",
   "metadata": {},
   "source": [
    "### *Анализ признаков*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "37bc0d98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='TIT', ylabel='Probability'>"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYkAAAEKCAYAAADn+anLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAeLklEQVR4nO3de5xU5Z3n8c+PFtKRy5hBMB2qO6BBuagQlgVHSRQQBS8wUZPIMG7WSFgSGeOqL6dnjQHWkCGu7GoSIsvghTBRFqMoKAi8IOjGxBEQVC5eCEG6gAmIV3SVi7/94xRYNHW6TlXX6arq+r5fr3p1nXOe55zf012v+vXznHOeY+6OiIhIJm2KHYCIiJQuJQkREQmlJCEiIqGUJEREJJSShIiIhFKSEBGRUCcUO4BCOvnkk7179+7FDkNEpGysW7fuLXfvEra9VSWJ7t27s3bt2mKHISJSNszszaa2xzrcZGYjzew1M9tqZvUZto8xs5fNbIOZrTWzIVHriohI/GJLEmZWBcwERgF9gLFm1qdRsZVAP3fvD3wXmJNDXRERiVmcPYlBwFZ33+buB4D5wJj0Au6+3z+bF6Q94FHriohI/OI8J9ENaEhbTgKDGxcys28A/wx0BS7NpW4UBw8eJJlM8vHHH+dTvexUV1eTSCRo27ZtsUMRkVYgziRhGdYdN5uguy8EFprZ14E7gAuj1gUwswnABIC6urrjtieTSTp27Ej37t0xy7Tb1sPd2bdvH8lkkh49ehQ7HBFpBeIcbkoCtWnLCWBXWGF3fxY4zcxOzqWuu89294HuPrBLl+Ov4vr444/p3Llzq08QAGZG586dK6bXJCLxizNJrAF6mlkPM2sHXA0sSi9gZl+x1Le3mQ0A2gH7otTNRSUkiCMqqa0iEr/YkoS7HwImAcuALcACd99kZhPNbGKq2JXARjPbQHA107c9kLFuXLHmwt0ZMmQIS5cuPbpuwYIFjBw5sohRiYjEw1rTQ4cGDhzojW+m27JlC7179y7ocTZu3Mg3v/lN1q9fz+HDh+nfvz9PP/00p512Ws77Onz4MFVVVQWNL442i0j8ErV17Ew2ZC/YSLdELcmGHXkd08zWufvA0O1KEvm59dZbad++PR9++CHt27fnzTff5JVXXuHQoUNMmTKFMWPGsH37dq655ho+/PBDAH75y19y7rnnsnr1aqZOnUpNTQ0bNmxg8+bNBY1NSUKkPJkZk2esyrne1JuHke93ebYk0aqm5WhJkydPZsCAAbRr147LLruMYcOGcf/99/Puu+8yaNAgLrzwQrp27cqKFSuorq7mjTfeYOzYsUenDXnhhRfYuHGjrkISkZKmJJGn9u3b8+1vf5sOHTqwYMECFi9ezF133QUEV1Tt2LGDL33pS0yaNIkNGzZQVVXF66+/frT+oEGDlCBEpOQpSTRDmzZtaNOmDe7Oo48+yhlnnHHM9ilTpnDKKafw0ksv8emnn1JdXX10W/v27Vs6XBGRnOl5EgVw8cUX84tf/OLomOD69esBeO+996ipqaFNmzbMmzePw4cPFzNMEZGcKUkUwO23387Bgwc5++yzOfPMM7n99tsB+MEPfsDcuXM555xzeP3119V7EJGyo6ubWqFKbLNIa1CKVzepJyEiIqGUJEREJJSShIiIhFKSEBGRUEoSIiISSklCRERCKUnErKGhgaFDh9K7d2/69u3LPffcA8Dbb7/NiBEj6NmzJyNGjOCdd94BYN++fQwdOpQOHTowadKkY/Z12223UVtbS4cOHVq8HSJSmSouSSRq6zCzgr0Stcc/MjXdCSecwIwZM9iyZQvPP/88M2fOZPPmzUyfPp3hw4fzxhtvMHz4cKZPnw4Ez6i+4447js4Dle7yyy/nhRdeiOX3IiKSScXN3bQz2ZDXzSphpt48rMntNTU11NTUANCxY0d69+7Nzp07eeKJJ1i9ejUA3/nOd7jgggv42c9+Rvv27RkyZAhbt249bl/nnHNOweIWEYmi4noSxbR9+3bWr1/P4MGD+ctf/nI0edTU1LBnz54iRycicjwliRayf/9+rrzySu6++246depU7HBERCJRkmgBBw8e5Morr2TcuHFcccUVAJxyyins3r0bgN27d9O1a9dihigikpGSRMzcneuuu47evXtz0003HV0/evRo5s6dC8DcuXMZM2ZMsUIUEQmlJBGz5557jnnz5rFq1Sr69+9P//79WbJkCfX19axYsYKePXuyYsUK6uvrj9bp3r07N910Ew8++CCJROLoM7BvvfVWEokEH330EYlEgilTphSpVSJSKSru6qZuidqsVyTlur+mDBkyJHQK35UrV2Zcv3379ozr77zzTu68886c4hMRaY6KSxLJhh3FDkFEpGxouElEREIpSYiISCglCRGR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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(\n",
    "             data=gas_train, x='TIT', hue='Year',\n",
    "             cbar=True,palette='dark', legend=True,\n",
    "             log_scale=True, bins=30,binwidth=0.002,\n",
    "             common_norm=False,stat='probability',\n",
    "             \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "a2c4246e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='AFDP', ylabel='Probability'>"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAY4AAAEJCAYAAACDscAcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAjMUlEQVR4nO3de5QU5bnv8e8DwmbLZRuJ6ISeEXRPBK/AmaDniDsCoqACMRojGmOISkgkxiXuhB00yjFkjxxJ5HgIbFQMuk0IiRpAIcqCEE9MjFzUBMELGmQaJ8LGezjI7Tl/VA1p2ma6iuma7p7+fdaaNd1V71v9dC3god5663nN3REREYmqXbEDEBGR8qLEISIisShxiIhILEocIiISixKHiIjEosQhIiKxHFbsAFrDJz/5Se/Vq1exwxARKStr1qz5L3c/Knt7RSSOXr16sXr16mKHISJSVszsjVzbNVQlIiKxKHGIiEgsShwiIhJLRdzjEBGJavfu3aTTaXbu3FnsUFpNp06dSKVSdOjQIVJ7JQ4RkQzpdJquXbvSq1cvzKzY4STO3dm+fTvpdJrevXtH6pPoUJWZDTezl81so5lNyrG/j5n9wcw+MrObMrafYGbPZ/y8b2Y3hPtuM7MtGfvOT/I7iEhl2blzJ927d6+IpAFgZnTv3j3WFVZiVxxm1h6YCQwD0sAqM1vk7uszmr0NXA98LrOvu78M9Ms4zhbg0YwmP3L3O5OKXUQqW6UkjSZxv2+SVxwDgY3u/rq77wLmA6MzG7j7VndfBexu5jhDgdfcPed8YhGRUubuDBo0iKVLl+7ftmDBAoYPH17EqFomycTRE2jIeJ8Ot8V1GfCzrG0TzOxPZjbXzD5xqAGKiCTNzJg9ezY33ngjO3fu5G9/+xuTJ09m5syZh3S8vXv3FjjC+JJMHLmufWItN2hmHYFRwC8yNs8CjicYymoEph+k7zgzW21mq7dt2xbnY0WK7tjqFGaW9+fY6lSxQ5UITj75ZEaOHMkdd9zBlClT+NKXvsTUqVP5zGc+Q//+/Vm4cCEAmzZt4qyzzmLAgAEMGDCA3//+9wCsXLmSwYMHc/nll3PKKacU86sAyc6qSgPVGe9TwJsxjzECWOvubzVtyHxtZvcAj+Xq6O5zgDkAdXV1Wh9Xysrm9BY23D81b7u+Yye3QjRSCLfeeisDBgygY8eOXHjhhQwZMoS5c+fy7rvvMnDgQM455xx69OjBsmXL6NSpE6+++ipjxozZXy7p2WefZd26dZFnPiUpycSxCqg1s94EN7cvAy6PeYwxZA1TmVmVuzeGby8C1rU0UBGRpHXu3JkvfvGLdOnShQULFrB48WLuvDOY47Nz5042b97Mpz71KSZMmMDzzz9P+/bteeWVV/b3HzhwYEkkDUgwcbj7HjObADwBtAfmuvuLZjY+3D/bzI4BVgPdgH3hlNsT3f19MzucYEbW17IOPc3M+hEMe23KsV9EpCS1a9eOdu3a4e48/PDDnHDCCQfsv+222zj66KN54YUX2LdvH506ddq/r3Pnzq0d7kEl+gCguy8BlmRtm53x+q8EQ1i5+u4AuufYfmWBwxQRaVXnnXced999N3fffTdmxnPPPUf//v157733SKVStGvXjnnz5pXEjfBcVKtKRKSV3XLLLezevZtTTz2Vk08+mVtuuQWAb3zjG8ybN48zzjiDV155paSuMjKZe9u/b1xXV+daj0PKiZlFvjleCX+HW9OGDRvo27dvscNodbm+t5mtcfe67La64hARkViUOEREJBYlDhERiUWJQ0REYlHiEBGRWJQ4REQkFiUOEZES09DQwODBg+nbty8nnXQSM2bMAODtt99m2LBh1NbWMmzYMN555x0Atm/fzuDBg+nSpQsTJkw44FiTJ0+murqaLl26FCw+JQ4RkWakqmsiVSqO+pOqrsn7mYcddhjTp09nw4YNPPPMM8ycOZP169dTX1/P0KFDefXVVxk6dCj19fVAsGb47bffvr/2VaaRI0fy7LPPFvScaM1xkTLWjuirt9WkevJGQzrZgNqgLekGbp2+omDHmzJxSN42VVVVVFVVAdC1a1f69u3Lli1bWLhwIStXrgTgqquu4uyzz+aOO+6gc+fODBo0iI0bN37sWGeccUbBYm+ixCFSxvZBpCfMQSXYy9WmTZt47rnnOP3003nrrbf2J5Sqqiq2bt1alJg0VCUiUqI+/PBDLr74Yu666y66detW7HD2U+IQESlBu3fv5uKLL+aKK67g85//PABHH300jY3BckSNjY306NGjKLEpcYiIlBh35+qrr6Zv377ceOON+7ePGjWKefPmATBv3jxGjx5dlPiUOERESszTTz/Ngw8+yIoVK+jXrx/9+vVjyZIlTJo0iWXLllFbW8uyZcuY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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(\n",
    "             data=gas_train, x='AFDP', hue='Year',\n",
    "             cbar=True,palette='dark', legend=True,\n",
    "             log_scale=True, bins=40,binwidth=0.02,\n",
    "             common_norm=False,stat='probability',\n",
    "             \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "id": "f3098ef6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='GTEP', ylabel='Probability'>"
      ]
     },
     "execution_count": 211,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(\n",
    "             data=gas_train, x='GTEP', hue='Year',\n",
    "             cbar=True,palette='dark', legend=True,\n",
    "             log_scale=True, bins=30,binwidth=0.02,\n",
    "             common_norm=False,stat='probability',\n",
    "             \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "id": "38a50f52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,4))\n",
    "fig =sns.histplot(\n",
    "             data=gas_train, x='TAT', hue='Year',\n",
    "             cbar=True,palette='dark', legend=True,\n",
    "             log_scale=True, bins=5,binwidth=0.001,\n",
    "             common_norm=False,stat='probability',\n",
    "             \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "id": "441e2943",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,4))\n",
    "fig =sns.histplot(\n",
    "             data=gas_train, x='TEY', hue='Year',\n",
    "             cbar=True,palette='dark', legend=True,\n",
    "             log_scale=True, bins=5,binwidth=0.01,\n",
    "             common_norm=False,stat='probability',\n",
    "             \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "id": "d43489b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(8,4))\n",
    "fig =sns.histplot(\n",
    "             data=gas_train, x='CDP', hue='Year',\n",
    "             cbar=True,palette='dark', legend=True,\n",
    "             log_scale=True, bins=5,binwidth=0.01,\n",
    "             common_norm=False,stat='probability',\n",
    "             \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 264,
   "id": "2605d417",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Year', ylabel='CO'>"
      ]
     },
     "execution_count": 264,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXgAAAEGCAYAAABvtY4XAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAOz0lEQVR4nO3df4xl5V3H8c9nZ3al/LCVzrUXWZbxD6riNoCOaFxjCpoWahF/pKStdjBp3D/aIpjqpDaK+CsxU9MQRWM2lMjaSq22hAoVxLJYdxXoLNLCdmlakWUZ9rIzEsLu4kJn9usf92yY2Z2dGeae596Z732/ksncOffc5/nenJnPPHnuc85xRAgAkM+6XhcAACiDgAeApAh4AEiKgAeApAh4AEhqsNcFzDU0NBTDw8O9LgMA1ozdu3dPR0RjoedWVcAPDw9rYmKi12UAwJphe9+pnmOKBgCSIuABICkCHgCSIuABICkCHgCSIuABICkCHgCSWlXr4AFgrrGxMbVarY7amJ6e1szMjAYHBzU0NLTidprNpsbHxzuqpduKBrztpyUdkjQraSYiRkr2ByCXVqulycnJWtqanZ2tra21ohsj+MsiYroL/QBIptlsdtxGq9XS7OysBgYGOmqvjlq6jSkaAKtWHVMio6OjmpycVLPZ1Pbt22uoau0o/SFrSPoX27ttb11oB9tbbU/YnpiamipcDgD0j9IBvyUifkTSlZI+bPunT9whIrZFxEhEjDQaC14QDQCwAkUDPiKeq74flHSnpEtL9gcAeE2xgLd9hu2zjj+W9A5JT5TqDwAwX8kPWd8i6U7bx/v5u4i4t2B/AIA5igV8RDwl6aJS7QMAFscySaS2ms6ElNbm2ZBYuwh4pMaZkOhnBDxSW01nQtZVD7BcBDxS40xI9DMuFwwASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJBU8YC3PWD7v2zfXbovAMBrujGCv17S3i70AwCYo2jA294o6eck3VqyHwDAyUqP4G+WNCbp2Kl2sL3V9oTtiampqcLlAED/KBbwtt8t6WBE7F5sv4jYFhEjETHSaDRKlQMAfafkCH6LpJ+3/bSkz0q63PanC/YHAJijWMBHxO9ExMaIGJb0XkkPRMSvluoPADAf6+ABIKnBbnQSEQ9KerAbfQEA2hjBA0BSBDwAJEXAA0BSBDwAJEXAA0BSBDwAJEXAA0BSBDwAJEXAA0BSBDwAJEXAA0BSBDwAJEXAA0BSBDwAJEXAA0BSXbke/Fo3NjamVqu14tdPT09rZmZGg4ODGhoa6qiWZrOp8fHxjtoA0B8I+GVotVqanJzsuJ3Z2dla2gGA5SDgl6HZbHb0+larpdnZWQ0MDHTcVqevB9A/CPhl6HRKZHR0VJOTk2o2m9q+fXtNVQHA4viQFQCSIuABICkCHgCSIuABICkCHgCSIuABICkCHgCSIuABICkCHgCSKhbwtk+z/Yjtr9neY/sPSvUFADhZyUsVvCLp8og4bHu9pJ22/zkiHirYJwCgUizgIyIkHa5+XF99Ran+AADzFZ2Dtz1g+zFJByXdHxEPL7DPVtsTtiempqZKlgMAfaVowEfEbERcLGmjpEttb15gn20RMRIRI41Go2Q5ANBXurKKJiJelPSgpCu60R8AoOwqmobtN1WP3yDpZyU9Wao/AMB8JVfRnCPpdtsDav8j+VxE3F2wPwDAHCVX0Xxd0iWl2gcALI4zWQEgKQIeAJIi4AEgKQIeAJIquYoGQEJb/mJLr0t4XTa8uEHrtE77X9y/Zmrfdd2uWtphBA8ASRHwAJAUAQ8ASRHwAJAUAQ8ASRHwAJDUsgPe9pm2zyhZDACgPksGvO0P2X5G0j5J+23vs/2h8qUBADqxaMDb/l1J75b09oh4c0ScLekySVdWzwEAVqmlRvAfkPRLEfHU8Q3V42skjZYsDADQmSWnaCLi6ALb/k/SsSIVAQBqsVTAP2v7Z07cWG07UKYkAEAdlrrY2G9Iusv2Tkm7JYWkH5O0RdLVhWsDAHRgqYB/RdKvSXqrpB+WZElfkfQpSSdN3QDL8cwfvq3XJbwuMy+cLWlQMy/sW1O1b7rx8V6XgB5bKuBvlvTxiLht7kbbI9VzV5UpCwDQqaXm4Ierm2fPExETkoaLVAQAqMVSAX/aIs+9oc5CAAD1Wirgv2r710/caPuDan/oCgBYpZaag79B0p22f0WvBfqIpA2SfrFgXQCADi0a8BHxvKSftH2ZpM3V5nsi4oHilQEAOrKsm25HxA5JOwrXAgCoEdeDB4CkCHgASKpYwNs+z/YO23tt77F9fam+AAAnW9Yc/ArNSPpoRDxq+yxJu23fHxHfKNgnAKBSbAQfEQci4tHq8SFJeyWdW6o/AMB8XZmDtz0s6RJJDy/w3FbbE7YnpqamulEOAPSF4gFv+0xJn5d0Q0S8dOLzEbEtIkYiYqTRaJQuBwD6RtGAt71e7XD/TER8oWRfAID5Sq6isdrXjd8bEZ8s1Q8AYGElR/Bb1L5p9+W2H6u+3lWwPwDAHMWWSUbETrXvAAUA6AHOZAWApAh4AEiKgAeApAh4AEiKgAeApAh4AEiKgAeApAh4AEiKgAeApAh4AEiq5B2divrR397e6xKW7azpQxqQ9Mz0oTVV9+5PjPa6BAAdYAQPAEkR8ACQFAEPAEkR8ACQFAEPAEkR8ACQFAEPAEkR8ACQFAEPAEkR8ACQFAEPAEkR8ACQFAEPAEkR8ACQFAEPAEkR8ACQFAEPAEkVC3jbt9k+aPuJUn0AAE6t5Aj+byRdUbB9AMAiigV8RHxF0gul2gcALK7nc/C2t9qesD0xNTXV63IAII2eB3xEbIuIkYgYaTQavS4HANLoecADAMog4AEgqZLLJO+Q9J+SfsD2s7Y/WKovAMDJBks1HBHvK9U2AGBpTNEAQFIEPAAkRcADQFIEPAAkRcADQFIEPAAkRcADQFIEPAAkRcADQFIEPAAkRcADQFIEPAAkRcADQFIEPAAkRcADQFIEPAAkRcADQFLF7ugEAJ1av2u9/LI7auP46/2yteH+DStuJ04PfWfLdzqqpdsIeACrll+21h2pZ6LBYfnIyv9ZHNOxWuroJgIewKoVp0fHweqjlo5JWifFadFRLWsNAQ9g1VprUyKrDR+yAkBSBDwAJEXAA0BSBDwAJEXAA0BSBDwAJEXAA0BSRQPe9hW2v2n727Y/VrIvAMB8xQLe9oCkv5R0paQLJb3P9oWl+gMAzFdyBH+ppG9HxFMR8aqkz0q6umB/AIA5Sl6q4FxJ++f8/KykHz9xJ9tbJW2VpE2bNi278d2fGO2wvO4ZHf1XTU6+pE1DZ2n7Gqq7lE03Pt7rEl6XwdFRaXJSg2efr003/luvy+m5Xdft6nUJWKaSI/iFLtt20tV6ImJbRIxExEij0ShYDgD0l5IB/6yk8+b8vFHScwX7AwDMUXKK5quSLrD9/ZImJb1X0vsL9gecZGxsTK1Wq6M2jr++1WppdLSzKbZms6nx8fGO2gCWq1jAR8SM7Y9Iuk/SgKTbImJPqf6AhbRaLU1OTtbS1uzsbG1tAd1Q9HrwEfElSV8q2QewmGaz2XEb09PTmpmZ0eDgoIaGhnpeD7Bc3PADqTEdgn7GpQoAICkCHgCSYopmGTpdicEqDAC9QMAvQ10rMViFAaCbCPhl6HTlA6swAPQCAb8MTIkAWIv4kBUAkiLgASApAh4AkiLgASApAh4AkiLgASApAh4AknLESXfR6xnbU5L29bqOQoYkTfe6CKwYx29ty3z8zo+IBe93uqoCPjPbExEx0us6sDIcv7WtX48fUzQAkBQBDwBJEfDds63XBaAjHL+1rS+PH3PwAJAUI3gASIqAB4CkCPgVsn2e7R2299reY/v6avvZtu+3/a3q+/dU299c7X/Y9i0ntPUntvfbPtyL99KP6jp+tk+3fY/tJ6t2/rRX76mf1Pz3d6/tr1Xt/LXtgV68pxII+JWbkfTRiPghST8h6cO2L5T0MUlfjogLJH25+lmSjkr6PUm/tUBb/yTp0vIlY446j9+fRcQPSrpE0hbbVxavHnUev2si4iJJmyU1JL2ndPHdQsCvUEQciIhHq8eHJO2VdK6kqyXdXu12u6RfqPY5EhE71f5FO7GthyLiQDfqRltdxy8iXo6IHdXjVyU9KmljN95DP6v57++l6uGgpA2S0qw8IeBrYHtY7dHbw5Lecjysq+/f28PSsAx1HT/bb5J0ldojR3RJHcfP9n2SDko6JOkfy1TafQR8h2yfKenzkm6YMxLAGlHX8bM9KOkOSX8eEU/VVR8WV9fxi4h3SjpH0ndJurym8nqOgO+A7fVq/3J9JiK+UG1+3vY51fPnqD0qwCpU8/HbJulbEXFz7YViQXX//UXEUUlfVHuaJwUCfoVsW9KnJO2NiE/OeeqLkq6tHl8r6a5u14al1Xn8bP+xpDdKuqHmMnEKdR0/22fO+YcwKOldkp6sv+Le4EzWFbL9U5L+XdLjko5Vmz+u9jzg5yRtkvSMpPdExAvVa56W9N1qf5DzoqR3RMQ3bI9Ler+k75P0nKRbI+Kmbr2XflTX8ZP0kqT9aofCK1U7t0TErd14H/2qxuP3v5LuVntqZkDSA5J+MyJmuvRWiiLgASAppmgAICkCHgCSIuABICkCHgCSIuABICkCHn3LbTvnXhzM9jW27+1lXUBdWCaJvmZ7s6R/UPtaJgOSHpN0RUT89wraGoiI2XorBFaOgEffq040OyLpjOr7+ZLepvbVBW+KiLuqC1r9bbWPJH0kIv7D9tsl/b6kA5IujogLu1s9cGoEPPqe7TPUvszvq2qf1bgnIj5dXR3yEbVH9yHpWEQctX2BpDsiYqQK+HskbY6I/+lF/cCpDPa6AKDXIuKI7b+XdFjSNZKusn38xhCnqX3a+3OSbrF9saRZSW+d08QjhDtWIwIeaDtWfVnSL0fEN+c+afsmSc9LukjtxQlzbxxxpEs1Aq8Lq2iA+e6TdF11tULZvqTa/kZJByLimKQPqP2BLLCqEfDAfH8kab2kr9t+ovpZkv5K0rW2H1J7eoZRO1Y9PmQFgKQYwQNAUgQ8ACRFwANAUgQ8ACRFwANAUgQ8ACRFwANAUv8PraFiAscFWNoAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(data=gas_train,x='Year',y='CO', ci='sd', capsize=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "aa868444",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "heatmap = plt.figure(figsize=(16, 6)) \n",
    "heatmap = sns.heatmap(gas_train.corr(),vmin=-1,vmax=1,annot=True,cmap='BrBG')\n",
    "heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62eb7600",
   "metadata": {},
   "source": [
    "### *Как видно из графиков, признаки имеют различные распределения и имеют также значительное количество выбросов*\n",
    "\n",
    "Для  решения этой проблемы мы нормализуем данные с помощью библиотеки AdjustedScaler. Суть метода заключается в вычислении границ “интервал доверия” с учетом асимметрии распределения, но чтобы для симметричного случая он был равен всё тому же 1,5 * IQR. Поиск подходящей формулы для определения границ “интервала доверия” производился с целью сделать долю, приходящуюся на выбросы, не превышающей такую же, как у нормального распределения и 1,5 * IQR — приблизительно 0,7%. Более подробно про этот метод и его эффективность лучше прочитать в  статье  Миа Хаберт и Елена Вандервирен (Mia Hubert and Ellen Vandervieren)  в статье “An Adjusted Boxplot for Skewed Distributions”. 2007 г."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "590d93fd",
   "metadata": {},
   "source": [
    "# *Расчет прогнозируемых значений СО*"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6c96568",
   "metadata": {},
   "source": [
    "# *Нормализация данных с помощью AdjustedScaler*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "40ae23b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from adjdatatools.preprocessing import AdjustedScaler\n",
    "new_scaler = AdjustedScaler()\n",
    "new_scaler.fit(gas_2012)\n",
    "gas_2011_scaled = new_scaler.transform(gas_2011)\n",
    "gas_2012_scaled = new_scaler.transform(gas_2012)\n",
    "gas_2013_scaled = new_scaler.transform(gas_2013)\n",
    "gas_scaled = pd.concat([gas_2011_scaled,gas_2012_scaled,gas_2013_scaled])\n",
    "X_scaled_CO = gas_scaled.drop(columns=['CO','NOX','AT','AP','AH','Year'])\n",
    "Y_scaled_CO = (gas_scaled['CO'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c5a3ba7",
   "metadata": {},
   "source": [
    "###  Создание базовой регрессионной модели"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "6c03cb20",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.00032998040490017644"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dummy = DummyRegressor(strategy='mean')\n",
    "dummy = dummy.fit(X_train, y_train)\n",
    "dummy.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2fabd8f3",
   "metadata": {},
   "source": [
    "### *Создание модели*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "7af959ff",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CV_score: 0.7274243694424355\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled_CO, Y_scaled_CO, random_state=42,test_size=0.1)\n",
    "adj_scal_RF = RandomForestRegressor(n_estimators=91,\n",
    "                                    criterion='mse',\n",
    "                                    max_features=1,\n",
    "                                    max_depth=15,\n",
    "                                    n_jobs=-1,ccp_alpha=0.0,\n",
    "                                    oob_score=True,\n",
    "                                    min_samples_leaf=5\n",
    "                                   )\n",
    "\n",
    "\n",
    "\n",
    "adj_scal_RF.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "y_pred_CO = adj_scal_RF.predict(X_test)\n",
    "\n",
    "print('CV_score:',cross_val_score(adj_scal_RF,X_train,y_train, scoring='r2').mean())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41a2d106",
   "metadata": {},
   "source": [
    "### *Определение важных признаков*"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6b58478",
   "metadata": {},
   "source": [
    "Как видно из рисунка такими признаками как AT,AH,AP,Year можно пренебречь."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "3320243d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " 1) TIT                            0.202508\n",
      " 2) CDP                            0.167537\n",
      " 3) TEY                            0.158034\n",
      " 4) GTEP                           0.152419\n",
      " 5) TAT                            0.095151\n",
      " 6) AFDP                           0.090212\n",
      " 7) AT                             0.044505\n",
      " 8) Year                           0.033020\n",
      " 9) AH                             0.029100\n",
      "10) AP                             0.027513\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "adj_scal_RF.fit(X_train,y_train)\n",
    "feat_labels = X_scaled_CO.columns[:]\n",
    "importances = adj_scal_RF.feature_importances_\n",
    "indices = np.argsort( importances )[ : : -1]\n",
    "for f in range(X_train.shape[1]):\n",
    "    print (\"%2d) %-*s %f\" % (f + 1, 30,\n",
    "            feat_labels[indices[f]],\n",
    "            importances[indices[f]]))\n",
    "plt.title ( 'Важность признаков' )\n",
    "plt.bar(range(X_train.shape[1]),\n",
    "              importances[indices],\n",
    "              align='center' )\n",
    "plt.xticks(range(X_train.shape[1]),\n",
    "            feat_labels[indices],rotation=90)\n",
    "plt.xlim([-1, X_train.shape[1]])\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "4e59582e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2: 0.5803192779581376\n"
     ]
    }
   ],
   "source": [
    "gas_2014_scaled = new_scaler.transform(gas_2014)\n",
    "gas_2015_scaled = new_scaler.transform(gas_2015)\n",
    "gas_scaled_pr = pd.concat([gas_2014_scaled,gas_2015_scaled])\n",
    "X_scaled_pr = gas_scaled_pr.drop(columns=['CO','NOX','AT','AP','AH'])\n",
    "Y_scaled_pr = (gas_scaled_pr['CO']) \n",
    "y_pred_RF = adj_scal_RF.predict(X_scaled_pr)\n",
    "print('R2:',r2_score(Y_scaled_pr,y_pred_RF))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76c5fd4d",
   "metadata": {},
   "source": [
    "# *Расчет прогнозируемых значений NOX* \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "id": "836c2229",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Year', ylabel='NOX'>"
      ]
     },
     "execution_count": 300,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(data=gas_train,x='Year',y='NOX', ci='sd', capsize=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "id": "8fa8d99d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_NOX = gas_scaled.drop(columns=['CO','NOX','Year','AP','AH'])\n",
    "Y_NOX = gas_scaled['NOX']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_NOX, Y_NOX, random_state=42,test_size=0.3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "194438fa",
   "metadata": {},
   "source": [
    "### *Проверка простейшей модели*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "a70ca284",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1.4256151396496719e-05"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dummy = DummyRegressor(strategy='mean')\n",
    "dummy = dummy.fit(X_train, y_train)\n",
    "dummy.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b1cc53d",
   "metadata": {},
   "source": [
    "#### *Создание модели и определение важных признаков*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "a5c97cf9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CV_R2: 0.7879873992316178\n"
     ]
    }
   ],
   "source": [
    "adj_scal_RF_NOX = RandomForestRegressor(n_estimators=90,\n",
    "                                    criterion='mse',\n",
    "                                    max_features=1,\n",
    "                                    max_depth=21,\n",
    "                                    n_jobs=-1,ccp_alpha=0.0,\n",
    "                                    oob_score=True,\n",
    "                                    min_samples_leaf=1\n",
    "                                    \n",
    "                                   )\n",
    "\n",
    "adj_scal_RF_NOX.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "adj_scal_RF_pred = adj_scal_RF_NOX.predict(X_test)\n",
    "\n",
    "\n",
    "print('CV_R2:',cross_val_score(adj_scal_RF_NOX,X_train,y_train, scoring='r2').mean())\n",
    "      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "374deda5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " 1) AT                             0.237603\n",
      " 2) TIT                            0.142111\n",
      " 3) GTEP                           0.111177\n",
      " 4) TEY                            0.100602\n",
      " 5) AFDP                           0.094243\n",
      " 6) CDP                            0.093218\n",
      " 7) TAT                            0.092837\n",
      " 8) AP                             0.072746\n",
      " 9) AH                             0.055462\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "feat_labels = X_NOX.columns[0:]\n",
    "adj_scal_RF_NOX.fit(X_train, y_train)\n",
    "importances = adj_scal_RF_NOX.feature_importances_\n",
    "indices = np.argsort( importances )[ : : -1]\n",
    "for f in range(X_train.shape[1]):\n",
    "    print (\"%2d) %-*s %f\" % (f + 1, 30,\n",
    "            feat_labels[indices[f]],\n",
    "            importances[indices[f]]))\n",
    "plt.title ( 'Важность признаков' )\n",
    "plt.bar(range(X_train.shape[1]),\n",
    "              importances[indices],\n",
    "              align='center' )\n",
    "plt.xticks(range(X_train.shape[1]),\n",
    "            feat_labels[indices],rotation=90)\n",
    "plt.xlim([-1, X_train.shape[1]])\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "753f0f9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = AdjustedScaler()\n",
    "scaler = scaler.fit(pd.concat([gas_2014,gas_2015]))\n",
    "gas_2014_scaled = scaler.transform(gas_2014)\n",
    "gas_2015_scaled = scaler.transform(gas_2015)\n",
    "gas_scaled_pr = pd.concat([gas_2014_scaled,gas_2015_scaled])\n",
    "X_scaled_pr_NOX = gas_scaled_pr.drop(columns=['CO','NOX','AP','AH'])\n",
    "Y_scaled_pr_NOX = (gas_scaled_pr['NOX'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "5078359a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2: 0.5752503661647861\n"
     ]
    }
   ],
   "source": [
    "y_pred_RF_NOX = adj_scal_RF_NOX.predict(X_scaled_pr_NOX)\n",
    "print('R2:',r2_score(Y_scaled_pr_NOX,y_pred_RF_NOX))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6d4442ef",
   "metadata": {},
   "source": [
    "### Обратное преобразование данных"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "5fb2a1a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_scaled_pr_NOX['NOX'] = y_pred_RF_NOX\n",
    "NOX_real = scaler.inverse_transform(X_scaled_pr_NOX)\n",
    "NOX_real = NOX_real.drop(['AT','TIT'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "7280957d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_scaled_pr['CO'] = y_pred_RF\n",
    "CO_real = scaler.inverse_transform(X_scaled_pr)\n",
    "CO_real = CO_real.drop(['AFDP','GTEP','TIT','TAT','TEY','CDP'],axis=1)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
