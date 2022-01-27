{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "LOOCV_and_ROC.py",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPwsRGxiqOARuVlZGtgYevU",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JingwenLi173/Python-for-data-analysis/blob/main/LOOCV_and_ROC_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9Zl0Et637dTI",
        "outputId": "730111f7-f18d-43c4-8414-2cf399cbbd50"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(150, 4)\n",
            "(150,)\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "from sklearn.ensemble import AdaBoostClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.model_selection import LeaveOneOut \n",
        "from sklearn import  datasets\n",
        "\n",
        "# load data\n",
        "iris = datasets.load_iris()\n",
        "X = iris.data\n",
        "y = iris.target\n",
        "print(X.shape)\n",
        "print(y.shape)\n",
        "# print(X,'\\n',y)\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# keep only two classes\n",
        "X, y = X[y != 2], y[y != 2]\n",
        "print(X.shape)\n",
        "print(y.shape)\n",
        "# print(X,'\\n',y)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y0HpUvfo7rE9",
        "outputId": "36f26357-3cce-4a13-c499-eee61a0e47b0"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(100, 4)\n",
            "(100,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "loo = LeaveOneOut()\n",
        "loo.get_n_splits(X)\n",
        "print(\"cross validation：\",loo.get_n_splits(X))\n",
        "\n",
        "y_pred = []\n",
        "for train_index, test_index in loo.split(X):\n",
        "    #print(\"train:\", train_index, \"TEST:\", test_index) \n",
        "    X_train, X_test = X[train_index], X[test_index]\n",
        "    y_train, y_test = y[train_index], y[test_index]\n",
        "    #print(X_train, X_test, y_train, y_test)\n",
        "    \n",
        "    # train model\n",
        "    model_bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = \"SAMME\", n_estimators = 10)\n",
        "    model_bdt.fit(X_train, y_train)\n",
        "   \n",
        "    # prediction\n",
        "    x_test_pred = model_bdt.predict(X_test)\n",
        "    \n",
        "    # print(x_test_pred)\n",
        "    y_pred.append(x_test_pred) # add prediction into table\n",
        "  \n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sLKkPNgd8YlV",
        "outputId": "bb7f3efd-6dad-4e9b-e77d-ed0225e728f3"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "cross validation： 100\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import roc_curve, auc\n",
        "\n",
        "y_pred = np.array(y_pred) # list to array\n",
        "print(y_pred.shape)\n",
        "fpr, tpr, threshold = roc_curve(y, y_pred)  # calculate false pos rate and true pos rate\n",
        "print(roc_curve(y, y_pred))\n",
        "print(fpr, tpr, threshold)\n",
        "roc_auc = auc(fpr, tpr)  # accuracy\n",
        "print(roc_auc)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u5m6C6zO9B2w",
        "outputId": "a8bc0ca6-3627-4d0e-c527-9bbe25bb0e91"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(100, 1)\n",
            "(array([0., 0., 1.]), array([0., 1., 1.]), array([2, 1, 0]))\n",
            "[0. 0. 1.] [0. 1. 1.] [2 1 0]\n",
            "1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "lw = 2 # line width\n",
        "plt.figure(figsize=(8, 5))\n",
        "plt.plot(fpr, tpr, color='darkorange',\n",
        "         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###false pos rate = x，true pos rate = y\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.title('Receiver operating characteristic example')\n",
        "plt.legend(loc=\"lower right\")\n",
        "plt.savefig('loocv.png',dpi=600) # save image\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "id": "yyHlkviYANH7",
        "outputId": "ec4a0646-419a-4d5f-eb62-550ba37ba318"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfkAAAFNCAYAAAAD7RaHAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd3gU5frG8e+TAgkQeidIDb0KgggqUgRF5agHUY69YaGICCIKoiACghSxHPmJHAt2UWyAICqCKL1LERFC7z2Q8v7+2CWuCEkI2UzK/bmuXOz0e5ZNnp133pkx5xwiIiKS84R4HUBERESCQ0VeREQkh1KRFxERyaFU5EVERHIoFXkREZEcSkVeREQkh1KRlyzPzFaZWUuvc3jNzF4zswGZvM1JZjYkM7cZLGb2HzObkc5lc+xn0MycmVX1OocEh+k6eTkXZrYJKAUkAkeAaUA359wRL3PlNGZ2J3Cvc66FxzkmAbHOuac8zjEIqOqcuzUTtjWJLLDPmcXMHBDjnNvgdRbJeDqSl/S41jlXAGgANASe8DjPOTOzsNy4bS/pPRfJfCrykm7OuR3AdHzFHgAzu9jM5pnZATNbFtjEaWZFzexNM9tmZvvN7LOAadeY2VL/cvPMrF7AtE1m1sbMyprZcTMrGjCtoZntMbNw//DdZrbGv/7pZlYhYF5nZg+b2Xpg/Zn2ycyu8zfNHjCz782s5mk5njCz1f71v2lmEeewD4+b2XLgqJmFmVk/M/vdzA7713m9f96awGtAMzM7YmYH/OOTm87NrKWZxZpZbzPbZWbbzeyugO0VM7MvzOyQmS0wsyFm9tPZ/i/NrEXA/9sWf0vCKUXM7Ct/zl/MrErAcmP98x8ys0VmdmnAtEFm9rGZvWNmh4A7zayJmf3s3852MxtvZnkClqltZt+a2T4z22lm/c2sPdAf6Ox/P5b55y1kZm/417PVv4+h/ml3mtlcMxttZnuBQf5xP/mnm3/aLn/2FWZWx8zuB/4D9PVv64uA/782/teh/lyn/u8WmVn5s7yvZ/x9MLNL/J/b8v7h+v7PVA3/8Bk/G2fYtwNmttG/vjv9/xe7zOyOgPknme9Uz7f+9f1gAb8Xp+XNa2YjzWyz//1/zcwiz/a5kWzAOacf/aT5B9gEtPG/jgZWAGP9w+WAvcDV+L5AtvUPl/BP/wr4ACgChAOX+8c3BHYBTYFQ4A7/dvKeYZvfAfcF5HkBeM3/uiOwAagJhAFPAfMC5nXAt0BRIPIM+1YNOOrPHQ709a8vT0COlUB5/zrmAkPOYR+W+peN9I/rBJT1v1ed/dsu4592J/DTafkmBWyvJZAAPOvPejVwDCjin/6+/ycfUAvYcvr6AtZbATgM3OJfVzGgQcA29wJN/O/pu8D7Acve6p8/DOgN7AAi/NMGAfHAv/z7GAk0Ai72z18RWAM84p8/CtjuX0+Ef7hpwLreOS33FOC/QH6gJPAr0DXg/UsAuvu3FRn4ngLtgEVAYcDwfWbKnP4+n+Vz3wff5766f9n6QLEzvK+p/T48h+/zHOlfX7eAZVP7bCQAd+H7rA0BNgMvA3mBK/3/nwUC9ucwcJl/+tjAzwK+34uq/tejgan4Pt9RwBfA817/3dFP+n88D6Cf7PXj/2N3xP9HwwGzgML+aY8Db582/3R8Ba8MkIS/CJ02z6vA4NPGreWvLwGBf2DvBb7zvzZ8xesy//A3wD0B6wjBV/gq+Icd0CqFfRsAfHja8luBlgE5HgiYfjXw+znsw92pvLdLgY7+13eSepE/DoQFTN+Fr4CG4iuu1QOmDTl9fQHTngCmnGXaJOD/Ttvn31LYh/1Aff/rQcCPqezzI6e2je9LxpKzzDeIgCKPr1/ICQK+rPmXnx3w/m0+bR3J7ynQCljnf79CzvY+n/a5P/UZXHvq/ymVfTvr74P/dTi+Lxor8PVtsXP4bKwPmFYX32e7VMC4vfz9i1rgF7MC+PrUlA/4vaiK7/fpKFAlYN5mwB+p7at+su6PmuslPf7lnIvCV2hqAMX94ysAnfxNiAf8zcwt8BX48sA+59z+M6yvAtD7tOXK4zuSOd0n+Jqxy+A7MkkC5gSsZ2zAOvbh+8NVLmD5LSnsV1ngz1MDzrkk//xnW/7PgIxp2Ye/bdvMbre/mvcPAHX4671Mi73OuYSA4WP4/oCXwHf0Gri9lPa7PPB7CtN3nGEbAJjZY+Y7PXLQvw+F+Ps+nL7P1czsSzPb4W/CHxowf2o5AlXAVyS3B7x//8V3RH/GbQdyzn0HjMd39LvLzF43s4Jp3HZac6b0+4BzLh5fAa4DjHL+qgpp+mzsDHh93L++08cVCBhOfi+cr5PsPv75+1UCX8vPooDtTvOPl2xKRV7SzTn3A74/UiP9o7bgO3IpHPCT3zk3zD+tqJkVPsOqtgDPnbZcPufce2fY5n5gBr4mzC74jlBcwHq6nraeSOfcvMBVpLBL2/D9YQZ8523x/UHfGjBP4LnXC/zLpHUfAv+IVwAmAN3wNfUWxncqwNKQMzW78TXnRp8l9+m2AFVSmH5G5jv/3he4CV8LTWHgIH/tA/xzP14FfsPXm7sgvnPtp+bfAlQ+y+ZOX88WfEfyxQPe74LOudopLPP3FTo3zjnXCN/pjGr4muFTXY60v18p/T5gZuWAp4E3gVFmltc/PrXPRnok//+bWQF8zfHbTptnD74vB7UD8hZyvk62kk2pyMv5GgO0NbP6wDvAtWbWzt85KcJ8HcSinXPb8TWnv2JmRcws3Mwu869jAvCAmTX1d4jKb2YdzCzqLNucDNwO/Nv/+pTXgCfMrDYkd8zqdA778iHQwcxam68jX298hSTwS8LDZhZtvs5/T+LrY5CefciPr5js9me9C9/R2ik7gWgL6JSWVs65ROBTfJ3N8vk7c92ewiLvAm3M7CbzdQgsZmYNUpj/lCh8XyZ2A2FmNhBI7Wg4CjgEHPHnejBg2pdAGTN7xN8BLMrMmvqn7QQqmlmIfx+34/uyN8rMCppZiJlVMbPL05AbM7vI/38Vjq+JOg5fq9CpbZ3tywbA/wGDzSzG/39dz8yKnWG+s/4++L9ATgLeAO7B1xdhsH+51D4b6XG1+TpX5vFvZ75z7m8tHf6WqwnAaDMr6d92OTNrd57bFg+pyMt5cc7tBt4CBvr/aHTEd3S2G9+RTB/++pzdhu9c8W/4zh8/4l/HQuA+fM2n+/F1drszhc1OBWKAHc65ZQFZpgDDgff9TcErgavOYV/W4utI9hK+o5pr8V0ueDJgtsn4istGfE22Q9KzD8651cAo4Gd8RaUuvo58p3wHrAJ2mNmetO5DgG74ms53AG8D7+H7wnKmLJvxnWvvja8Zdym+zmSpmY6vOXcdvlMXcaR8WgDgMXwtMIfxFZRTX5Jwzh3G1zntWn/u9cAV/skf+f/da2aL/a9vB/IAq/G95x/jbwpPg4L+7e/3Z9+LrxMn+ApvLX+T9WdnWPZFfF8IZ+D7wvIGvs5zf5PK70MPfKcWBvhbou4C7jKzS9Pw2UiPyfhaDfbh6/x4tvsNPI7vszvf/zs0E18HQ8mmdDMckTQy342A7nXOzfQ6y7kys+FAaefcHanOLDmK5bKb+8jf6UheJAcysxr+ZmQzsyb4moSneJ1LRDKX7gIlkjNF4WuiL4uvyXcU8LmniUQk06m5XkREJIdSc72IiEgOpSIvIiKSQ2W7c/LFixd3FStW9DqGiIhIpli0aNEe51y67jyY7Yp8xYoVWbhwodcxREREMoWZ/Zn6XGem5noREZEcSkVeREQkh1KRFxERyaFU5EVERHIoFXkREZEcSkVeREQkh1KRFxERyaGCVuTNbKKZ7TKzlWeZbmY2zsw2mNlyM7swWFlERERyo2AeyU8C2qcw/Sogxv9zP/BqELOIiIjkOkEr8s65H4F9KczSEXjL+cwHCptZmWDlERERyW28vK1tOWBLwHCsf9z2FJfauQhGWRBjiYiIeG/LgYI8/lXb81pHtrh3vZndj69Jn0bRHocREREJouPxYbwwuznDZrfgeHw48Em61+Vlkd8KlA8YjvaP+wfn3OvA6wCNy5ujtwt+OhERkUzknOOTT9bw2GMz+PPPgwB06lSLjz5K/zq9LPJTgW5m9j7QFDjonEu5qV5ERCQH2rPnGJ06fcT3328CoH79Uowd257LL6+InccZ6qAVeTN7D2gJFDezWOBpIBzAOfca8DVwNbABOAbcFawsIiIiWVmRIhEcPBhH8eL5eO65VtxzT0NCQ8+/b7w5l72avhuXN7dwS/bKLCIiEig+PpFXX13I9dfXoHz5QgCsXbuHkiXzU6RI5N/mNbNFzrnG6dlOtuh4JyIiklPMmPE7jzwyjTVr9jB/fiyTJ98IQPXqxTN8WyryIiIimWDDhn307j2DqVPXAlClShFuuaVOULepIi8iIhJEhw+fYMiQHxk9ej7x8UkUKJCHAQMuo2fPpuTNG9wyrCIvIiISRBs37mfkyJ9JSnLceWcDhg5tRZkyUZmybRV5ERGRDLZ69W5q1SoBQP36pRk5si3Nm19AkyblMjWHHjUrIiKSQbZtO8ztt0+hdu1X+Oqrdcnje/VqlukFHnQkLyIict7i4hIYPfpnnntuDkePxpMnTygbN+73OpaKvIiISHo55/j887X07j0juahff30NRo68ksqVi3icTkVeREQk3d54Ywn33fcFAHXqlGTMmHa0bl3Z41R/0Tl5ERGRcxB4p9ibb65DnTolGT/+KpYs6ZqlCjzoSF5ERCRNEhKSeP31Rbz22kLmzbuHAgXyUKBAHpYte4CQkPN4ikwQ6UheREQkFd999wcNG/6Xhx/+mhUrdjF58orkaVm1wIOO5EVERM7qjz/289hj3/Lpp2sAqFixMKNGXcn119fwOFnaqMiLiIicwcsv/0rv3jM4cSKRfPnCefLJS3n00WZERGSf0pl9koqIiGSiChUKc+JEIv/5T12GD29DuXIFvY50zlTkRUREgIULtzF37mZ69rwYgA4dYlix4kHq1CnpcbL0U5EXEZFcbceOI/TvP4s331xKSIhxxRWVqFevFGaWrQs8qMiLiEgudfJkImPHzmfw4B85fPgk4eEh9Op1MRUrFvY6WoZRkRcRkVznq6/W0avXdNav3wfAtddWY9SoK4mJKeZxsoylIi8iIrnOu++uYP36fdSsWZzRo9vRrl1VryMFhYq8iIjkeAcOxLFr11GqVfMdqQ8f3oamTcvx0EMXER4e6nG64NEd70REJMdKTPTdijYm5iVuvvljEhOTAChfvhA9e16cows8qMiLiEgO9eOPf9K48QS6dv2SPXuOUaBAHvbtO+51rEyl5noREclRNm8+SJ8+3/Lhh6sAKF++ICNHXkmnTrUwy7r3mQ8GFXkREckx4uMTueSSN9i69TCRkWE8/nhz+vRpTr584V5H84SKvIiIZGvOOZKSHKGhIYSHh9KvXwt++mkzI0a05YILCnkdz1PmnPM6wzlpXN7cwi3ZK7OIiATHkiXb6dlzGtdeW40+fZoDvqKfk5rlzWyRc65xepZVxzsREcl2du8+SteuX9Co0evMmbOZ115bREKCr+d8Tirw50tFXkREso34+ETGjJlPTMxLvP76YkJDQ3jkkaYsWnQ/YWEqaafTOXkREckWtm8/TKtWb/Hbb3sAaNeuCqNHt6NmzRIeJ8u6VORFRCRbKF26AEWKRFC1alFGj25Hhw4xappPhYq8iIhkSYcOneC5537k/vsbUaVKUcyMDz/sRIkS+cibV+UrLfQuiYhIlpKU5Pjf/5byxBOz2LnzKOvW7WPKlM4AREcX9Dhd9qIiLyIiWca8eVvo0eMbFi3aDkCzZtH079/C41TZl4q8iIh4btu2w/Tt+y3vvrsCgLJloxgxog1dutTVeffzoCIvIiKeO3ToBB98sIq8eUN57LFL6NevBQUK5PE6VranIi8iIpnOOcfs2Zu44oqKmBk1ahRn4sTraNHiAipVKuJ1vBxDdw4QEZFMtWLFTtq0eZvWrd/ik0/WJI+/7bb6KvAZTEfyIiKSKfbuPcbAgbN57bVFJCU5ihaN5OTJRK9j5Wgq8iIiElQJCUm89tpCBg6czf79cYSGGt26XcQzz1xB0aKRXsfL0VTkRUQkqP7734V07/4NAK1bV2LMmPbUqVPS41S5g4q8iIhkuLi4BCIifCXm7rsb8umnv9G9exM6dqyuS+IykTreiYhIhjly5CT9+88iJuYlDhyIAyAyMpxZs27nX/+qoQKfyYJa5M2svZmtNbMNZtbvDNMvMLPZZrbEzJab2dXBzCMiIsGRlOR4++1lVKv2Es8//xOxsYf46qt1XsfK9YLWXG9mocDLQFsgFlhgZlOdc6sDZnsK+NA596qZ1QK+BioGK5OIiGS8X3/dSs+e05g/PxaAJk3KMXZsey6+ONrjZBLMc/JNgA3OuY0AZvY+0BEILPIOOPW0gULAtiDmERGRDDZ48A8MHPg94HsU7LBhrbnttvqEhKhZPisIZpEvB2wJGI4Fmp42zyBghpl1B/IDbYKYR0REMtgll5QnT55QevW6mCefvJSoqLxeR5IAXne8uwWY5JyLBq4G3jazf2Qys/vNbKGZLcz0hCIiAvhuRfvFF2sZOHB28rjWrSuzaVNPhg1rowKfBQXzSH4rUD5gONo/LtA9QHsA59zPZhYBFAd2Bc7knHsdeB2gcXlzwQosIiJntmbNbh55ZDozZvwOwL/+VYMLLywDQJkyUV5GkxQE80h+ARBjZpXMLA9wMzD1tHk2A60BzKwmEAHsDmImERE5B/v3H+eRR6ZRt+6rzJjxO4UK5WXMmHbUraub2WQHQTuSd84lmFk3YDoQCkx0zq0ys2eBhc65qUBvYIKZ9cLXCe9O55yO1EVEsoAJExbRv/937NlzjJAQ44EHGvHss1dQokR+r6NJGgX1jnfOua/xXRYXOG5gwOvVQPNgZhARkfSZPz+WPXuOcdllFRg7tj0NGpT2OpKcI93WVkREAPjzzwPs3Xs8+Vz70KGtadeuKp061dKd6rIpr3vXi4iIx44ePcnTT8+mRo2Xue22KcTH+x7/WqpUAW66qbYKfDamI3kRkVzKOccHH6yiT59viY09BED9+qU4ejSewoVDPU4nGUFFXkQkF1q8eDs9enzD3Lm+e5Y1bFiaceOuokWLCzxOJhlJRV5EJJc5cSKBDh0ms2PHEUqUyMfQoa25664GhIbqDG5OoyIvIpILnDyZSFKSIyIijLx5wxg2rDXLl+9k4MDLKVQowut4EiT62iYiksN988166tV7leHDf0oed8cdDRg1qp0KfA6nIi8ikkOtW7eXDh0mc/XVk1m7di+ffbaWxMQkr2NJJlJzvYhIDnPwYByDB//I2LG/kJCQRMGCeXn66cvp1q2JzrvnMiryIiI5yJYtB2nceAK7dh3FDO69tyFDhrSiVKkCXkcTD6jIi4jkINHRBalTpyQnTiQwdmx7GjUq63Uk8ZDabUREsrHY2EPceuunrF7te4CnmfHxx52YM+cuFXjRkbyISHZ0/Hg8I0fOY9iwuRw7Fs+hQyeYOvUWAIoUifQ4nWQVKvIiItmIc45PPlnDY4/N4M8/DwLQqVMtRoxo63EyyYpU5EVEson16/dy//1f8v33mwCoV68UY8e2p2XLip7mkqxLRV5EJJsICwvh55+3UKxYJEOGtOLeey8kLExdq+TsVORFRLKo+PhEPvpoNTffXIeQEKNSpSJ88slNNGtWnqJFdd5dUqciLyKSBc2cuZGePaexevVuEhKSuP32+gB06FDN42SSnajIi4hkIb//vo/evWfw+edrAahcuQglS+b3OJVkVyryIiJZwOHDJxg6dA4vvjifkycTyZ8/nKeeuoxevS4mb179qZb00SdHRCQLmDRpKcOGzQXg9tvr8/zzrSlbNsrjVJLdqciLiHhk795jFCuWD4CuXRvz88+x9OzZlKZNoz1OJjlFmq+9MLN8wQwiIpJbbNt2mDvu+IyYmJfYvfsoAHnyhDJ58o0q8JKhUi3yZnaJma0GfvMP1zezV4KeTEQkh4mLS2DYsJ+oVu0l3nprGUePxjN37havY0kOlpbm+tFAO2AqgHNumZldFtRUIiI5iHOOqVPX8uijM9i4cT8A//pXDUaObEuVKkU9Tic5WZrOyTvntphZ4KjE4MQREcl5Hn10OmPG/AJArVolGDu2PW3aVPY4leQGaTknv8XMLgGcmYWb2WPAmiDnEhHJMTp1qk2RIhGMG9eeZcseUIGXTGPOuZRnMCsOjAXaAAbMAHo45/YFP94/NS5vbuGWlDOLiHglISGJCRMWsXz5Tl599Zrk8UePniR//jweJpPsyswWOecap2fZtDTXV3fO/ee0DTYH5qZngyIiOdXs2X/wyCPTWb58JwD33nshjRqVBVCBF0+kpbn+pTSOExHJlTZtOsC///0hrVq9xfLlO6lQoRAff9yJCy8s43U0yeXOeiRvZs2AS4ASZvZowKSCQGiwg4mIZAfPPPM9zz//EydOJJIvXzhPPNGC3r2bERkZ7nU0kRSb6/MABfzzBN5b8RDw72CGEhHJLvbsOcaJE4l06VKX4cPbEB1d0OtIIsnS0vGugnPuz0zKkyp1vBMRLy1atI2jR+O57LIKAOzbd5w1a3bTvPkFHieTnCrYHe+OmdkLQG0g4tRI51yr9GxQRCQ72rnzCP37z+LNN5dSuXIRVq16iLx5wyhaNFIFXrKstHS8exffLW0rAc8Am4AFQcwkIpJlnDyZyMiR84iJeYmJE5cSFhbC9dfXICEhyetoIqlKy5F8MefcG2bW0zn3A/CDmanIi0iO99VX6+jVazrr1/tuC9KhQwwvvtiOatWKeZxMJG3SUuTj/f9uN7MOwDZAN1sWkRwtLi6Brl2/ZOvWw1SvXozRo9tx1VUxXscSOSdpKfJDzKwQ0Bvf9fEFgUeCmkpExAMHD8YRGhpCgQJ5iIgIY+zY9vz550G6dWtCnjy6cliyn1R7159xIbPmzjlP7nin3vUiktESE5N4882l9O8/i7vvbsiwYW28jiSSLCi9680sFLgJKAdMc86tNLNrgP5AJNAwPRsUEclKfvppMz17TmPx4u0ALFy4jaQkR0iIpbKkSNaXUnP9G0B54FdgnJltAxoD/Zxzn2VGOBGRYNmy5SB9+87k/fdXAhAdXZAXXmhL5861Oe3R2iLZVkpFvjFQzzmXZGYRwA6ginNub+ZEExEJjj/+2E/t2q9w/HgCERFh9O17CX37NtdDZCTHSanIn3TOJQE45+LMbKMKvIjkBJUqFaFVq0rkz5+HESPaUKFCYa8jiQRFSjfDqWFmy/0/KwKGV5jZ8rSs3Mzam9laM9tgZv3OMs9NZrbazFaZ2eT07ISISEqWLdtB69ZvsXTpjuRxn3xyEx988G8VeMnRUjqSr3k+K/Z33HsZaAvEAgvMbKpzbnXAPDHAE0Bz59x+Myt5PtsUEQm0e/dRBgyYzYQJi0lKcgwa9D2ffXYzAHnzpuUKYpHs7ayf8gx4KE0TYINzbiOAmb0PdARWB8xzH/Cyc26/f5u7znObIiLExyfyyisLGDToBw4ciCMsLIQePZowcODlXkcTyVTB/CpbDtgSMBwLND1tnmoAZjYX3zPqBznnpp2+IjO7H7gfoFF0ULKKSA6xdOkOunT5hDVr9gBw5ZVVGDOmHTVrlvA4mUjm87q9KgyIAVoC0cCPZlbXOXcgcCbn3OvA6+C7GU5mhxSR7KNEiXz8+edBqlYtyosvXsk111TTJXGSa6XlKXSYWaSZVT/HdW/Fd539KdH+cYFiganOuXjn3B/AOnxFX0QkTQ4fPsGoUfNITPQ9Fa5cuYLMnHkbK1c+yLXXVleBl1wt1SJvZtcCS4Fp/uEGZjY1DeteAMSYWSUzywPcDJy+3Gf4juIxs+L4mu83pjm9iORaSUmOSZOWUq3aeB577FsmTFicPK1Zs/LqWCdC2prrB+HrRPc9gHNuqZlVSm0h51yCmXUDpuM73z7RObfKzJ4FFjrnpvqnXWlmq4FEoI+uxReR1MyfH0uPHt+wYME2AC6+OJrGjct6nEok60nTo2adcwdPa/JK03lx59zXwNenjRsY8NoBj/p/RERStHXrIfr1m8U77/hu1VG2bBTDh7ehS5e6ute8yBmkpcivMrMuQKj/uvYewLzgxhIR+aepU9fyzjvLyZs3lN69m/HEE5dSoIBuRStyNmkp8t2BJ4ETwGR8TexDghlKRATAOcfvv++natWiANx3XyPWr99Ht25NqFy5iMfpRLK+VJ8nb2YXOucWpzhTJtLz5EVyh5Urd9Gz5zR++SWWdeu6U7ZslNeRRDxxPs+TT8sldKPMbI2ZDTazOunZiIhIWu3bd5xu3b6mfv3X+O67P8iTJ5TVq3d7HUskW0q1yDvnrgCuAHYD//U/oOapoCcTkVwlISGJV15ZQEzMS7z88gIAHn74Itav706bNpU9TieSPaXaXP+3mc3qAn2Bzs45T3q7qLleJGe6557PmThxKQCtWlVizJh21K1byuNUIt4LanO9mdU0s0H+x82+hK9nve4gLyLnLfAg46GHLqJy5SJ88slNzJx5mwq8SAZIS8e7n4EPgA+dc9syJVUKdCQvkv0dPXqS55//iXXr9vLhh52SxycmJhEamqa7bYvkGudzJJ/qJXTOuWbpWbGIyOmcc0yevILHH5/J1q2HAVi+fCf16vmO2lXgRTLWWYu8mX3onLvJ30wfeOhs+G5WVy/o6UQkx1i4cBs9enzDzz/HAtC4cVnGjm2fXOBFJOOldCTf0//vNZkRRERyJuccDz74Ff/97yIASpXKz/PPt+aOOxroVrQiQXbWtjHn3Hb/y4ecc38G/gAPZU48EcnuzIyoqDyEh4fQt+8lrFvXnbvuaqgCL5IJ0tLxbrFz7sLTxi33qrleHe9EsjbnHF99tZ6QEOPqq2MAOHToBDt3HiEmppjH6USyn6B0vDOzB/EdsVc2s+UBk6KAuenZmIjkbGvW7KZXr+lMn/475csX5LffupEvXzgFC+alYMG8XscTyXVSOic/GfgGeB7oFzD+sHNuX1BTiUi2cuBAHM888z3jx8O0VXAAACAASURBVC8gISGJQoXy8uijzQgPV295ES+lVOSdc26TmT18+gQzK6pCLyKJiUm88cYSnnzyO/bsOYYZdO3aiMGDr6BEifxexxPJ9VI7kr8GWITvErrAXjIO0M2kRXK5+Pgkhg37iT17jnHZZRUYO7Y9DRqU9jqWiPidtcg7567x/1sp8+KISFa3efNBoqLyUKRIJBERYbz6agcOHjxBp061MFOPeZGsJC33rm9uZvn9r281sxfN7ILgRxORrOTYsXgGDfqe6tXH8/TT3yePb9euKjfdVFsFXiQLSkuvmFeBY2ZWH+gN/A68HdRUIpJlOOf44IOV1Kgxnmee+YG4uAT27TvOuTzBUkS8keq964EE55wzs47AeOfcG2Z2T7CDiYj3lizZTs+e05gzZzMADRqUZty49lx6aQWPk4lIWqSlyB82syeA24BLzSwECA9uLBHx2vr1e2nceAJJSY7ixfMxdGgr7r67oR4iI5KNpKXIdwa6AHc753b4z8e/ENxYIuKFpCSXfLvZmJhidO5cm9KlCzBw4OUULhzhcToROVep3tYWwMxKARf5B391zu0KaqoU6La2IsExbdoGHn10OhMnduTii6MB3/l4dagT8db53NY2Lb3rbwJ+BToBNwG/mNm/07MxEcl61q/fyzXXTOaqq95lzZo9jBkzP3maCrxI9paW5vongYtOHb2bWQlgJvBxMIOJSHAdOnSCIUN+ZMyY+cTHJxEVlYeBAy+nR4+mXkcTkQySliIfclrz/F7SdumdiGRRc+du5sYbP2TnzqOYwd13N+C551pTunQBr6OJSAZKS5GfZmbTgff8w52Br4MXSUSCLSamGHFxCTRrFs24cVfRuHFZryOJSBCktePdDUAL/+Ac59yUoKZKgTreiZy7rVsPMWbMfIYObU14eCgAa9fuoVq1YjrvLpLFBet58jHASKAKsAJ4zDm3NX0RRcQLcXEJjBo1j6FDf+LYsXjKlo2iV69mAFSvXtzjdCISbCk1108E3gJ+BK4FXgJuyIxQInJ+nHNMmfIbvXvPYNOmAwDccENN/vWvGh4nE5HMlFKRj3LOTfC/XmtmizMjkIicn5Urd9GjxzfMnr0JgDp1SjJ2bHtatdIDJUVym5SKfISZNeSv58hHBg4751T0RbKgBQu2Mnv2JooWjWTw4Cu4//5GhIXpghiR3CilIr8deDFgeEfAsANaBSuUiKRdQkISixdvp0mTcgDccUcDdu06yn33NaJo0UiP04mIl9LUuz4rUe96kb/MmrWRnj2n8fvv+1mz5mEqVizsdSQRyWBBva2tiGQ9Gzfu54YbPqBNm7dZtWo3ZcoUYMeOI17HEpEsJi03wxGRLOLIkZMMHTqHUaN+5uTJRPLnD+fJJy+lV69mRETo11lE/k5/FUSyka5dv2Ty5BUA3HZbPYYNa0PZslEepxKRrCrVIm++22H9B6jsnHvW/zz50s65X4OeTkSIj09Mvkvdk09eyqZNBxg16srkx8GKiJxNqh3vzOxVIAlo5ZyraWZFgBnOuYtSXDBI1PFOcovt2w/zxBOz2LbtMNOn36rbz4rkUkG5rW2Aps65C81sCYBzbr+Z5UnPxkQkdSdOJDBmzHyGDJnDkSMnyZMnlDVr9lCrVgmvo4lINpOWIh9vZqH4ro0/9Tz5pKCmEsmFnHN88cU6Hn10Or//vh+A666rzqhRV1K1alGP04lIdpSWS+jGAVOAkmb2HPATMDQtKzez9ma21sw2mFm/FOa70cycmaWrOUIku3POcf31H9Cx4/v8/vt+atYszvTpt/L55zerwItIuqV6JO+ce9fMFgGt8d3S9l/OuTWpLec/+n8ZaAvEAgvMbKpzbvVp80UBPYFf0pFfJEcwM+rXL8UPP/zJM8+05MEHGyd3thMRSa+0dLy74EzjnXObU1muGTDIOdfOP/yEf7nnT5tvDPAt0Aff42wXprRedbyTnCAxMYn/+7/FFCoUwc031wHg2LF4jh2Lp3jxfB6nE5GsJNgd777Cdz7egAigErAWqJ3KcuWALQHDsUDTwBnM7EKgvHPuKzPrk9bQItnZDz9somfPaSxbtpNSpfJzzTXVKFAgD/nyhZMvX7jX8UQkB0lLc33dwGF/YX7ofDdsZiH4HnhzZxrmvR+4H6CRLg2WbOrPPw/Qp8+3fPSR74xVhQqFGDnySvLnV2EXkeA45zveOecWm1nT1OdkK1A+YDjaP+6UKKAO8L3/+t/SwFQzu+70Jnvn3OvA6+Brrj/XzCJeiotL4Pnn5zBixDzi4hKIjAzjiSda8NhjlxAZqQIvIsGTljvePRowGAJcCGxLw7oXADFmVglfcb8Z6HJqonPuIFA8YDvfk4Zz8iLZTWio8eGHq4mLS+CWW+owfHgbypcv5HUsEckF0nIkH3hj7AR85+g/SW0h51yCmXUDpgOhwETn3CozexZY6Jybmp7AItnB4sXbiY4uSMmS+QkPD2XChGsBaNHijP1YRUSCIsXe9f7L4IY75x7LvEgpU+96ycp27TrKk0/O4o03lnDPPQ2ZMOE6ryOJSDYXlN71ZhbmPxpvnv5oIrnDyZOJjB//K8888wOHDp0gLCyEwoUjcM7pnvMi4pmUmut/xXf+famZTQU+Ao6emuic+zTI2USyhW++WU+vXtNZu3YvAFddVZXRo9tRvXrxVJYUEQmutJyTjwD2Aq3463p5B6jIS663Zs1urr56MgDVqhVj9Oh2XH11jMepRER8UiryJf0961fyV3E/RSfFJdc6fjw++dK3mjVL8PDDF1GpUmG6d29Knjy6Fa2IZB0pPaAmFCjg/4kKeH3qRyRXSUpyvPHGYipWHMvs2X8kjx8//mp6975EBV5EspyUjuS3O+eezbQkIlnY3Lmb6dlzGosWbQdg8uQVXHFFJY9TiYikLKUiry7BkuvFxh6ib99vee+9lQBERxdkxIg2yQ+VERHJylIq8q0zLYVIFjRjxu9cf/0HHDsWT0REGH36XMLjjzcnf/48XkcTEUmTsxZ559y+zAwiktVcdFFZ8uULp0OHGEaMaEvFioW9jiQick5S6ngnkqssX76T226bQlxcAgBFikSyatVDfPhhJxV4EcmWzvkpdCI5zZ49xxgw4Dtef30xSUmOunVL0rev70aPJUvm9zidiEj6qchLrhUfn8irry7k6ae/58CBOEJDjR49mnDffRd6HU1EJEOoyEuu9MMPm3jooa9ZvXo3AG3bVmbMmPbUqlXC42QiIhlHRV5ypR07jrB69W6qVCnCiy+249prq+lBMiKS46jIS65w+PAJ5szZnHxf+Ztuqs2JE4l07lybvHn1ayAiOZN610uOlpTkeOutZVSvPp6OHd9n7do9AJgZt99eXwVeRHI0/YWTHOuXX2Lp0WMav/66FYAmTcoRH5/kcSoRkcyjIi85zvbth+nXbxZvvbUMgNKlCzB8eBtuvbUeISE67y4iuYeKvOQ4vXpN54MPVpEnTyiPPnox/ftfSlRUXq9jiYhkOhV5yfaccxw6dIJChSIAGDKkFQkJSQwf3oYqVYp6nE5ExDvmnPM6wzlpXN7cwi3ZK7MEz6pVu+jVazrHjsUzZ85dugxORHIcM1vknGucnmV1JC/Z0v79x3n66e955ZUFJCY6CheOYMOGfcTEFPM6mohIlqEiL9lKQkISEyYsYsCA2ezde5yQEOPBBxvz7LNXULx4Pq/jiYhkKSrykm0452jZchJz524BoGXLiowd25569Up5nExEJGtSkZdsw8y45ppqxMYeYtSoK7nhhpo6By8ikgJ1vJMs6+jRkwwfPpfy5Qty332NADhxIoGkJEdkZLjH6UREMoc63kmO4pzjvfdW0rfvt2zdepiiRSPp0qUu+fPn0W1oRUTOgf5iSpayaNE2evSYxrx5vvPuF15YhnHj2pM/fx6Pk4mIZD8q8pIlHDlykkcemcbEiUtwDkqWzM/zz7fmzjsb6Fa0IiLppCIvWUJkZBgLF24jLCyEnj2b8tRTlyXfwU5ERNJHRV4889VX66hfvzTR0QUJDQ3hzTc7kj9/HqpV0w1tREQygp4nL5nut9/2cPXV73LNNe/Rr9/M5PENG5ZRgRcRyUA6kpdMc+BAHM8++wMvvfQrCQlJFCyYl8aNy+Kc0/XuIiJBoCIvQZeYmMTEiUt48snv2L37GGZw330XMmRIK0qWzO91PBGRHEtFXoLut9/20LXrlzgHl156AWPHtqdhwzJexxIRyfFU5CUodu06mnyUXrt2SQYMuIxatUpw00211TQvIpJJ1PFOMtSxY/E8++wPVKw4hq+/Xp88/plnrqBz5zoq8CIimUhFXjKEc44PP1xFzZov8/TT33P8eALff7/J61giIrmamuvlvC1btoOePafxww9/AlC/finGjm3P5ZdX9DaYiEgupyIv52XKlDX8+98fkZTkKFYskueea8W9915IaKgaiUREvKYiL+elTZvKlCsXxQ031OTppy+nSJFIryOJiIifiryckxkzfueFF+YxZUpnChTIQ1RUXn77rRv58un57iIiWY3aVCVNNmzYR8eO79Ou3TvMnLmRV19dkDxNBV5EJGsKapE3s/ZmttbMNphZvzNMf9TMVpvZcjObZWYVgplHzt3hwyd4/PFvqVXrZaZOXUtUVB5GjGhDjx5NvY4mIiKpCFpzvZmFAi8DbYFYYIGZTXXOrQ6YbQnQ2Dl3zMweBEYAnYOVSc7NZ5/9xoMPfsWOHUcAuOuuBgwd2prSpQt4nExERNIimOfkmwAbnHMbAczsfaAjkFzknXOzA+afD9waxDxyjkJDjR07jnDxxdGMG9eeiy4q53UkERE5B8Fsri8HbAkYjvWPO5t7gG+CmEdSsW3bYd5+e1ny8DXXVGPatP8wd+7dKvAiItlQluhdb2a3Ao2By88y/X7gfoBG0ZkYLJeIi0vgxRd/ZujQORw/nkCDBqWpW7cUZka7dlW9jiciIukUzCK/FSgfMBztH/c3ZtYGeBK43Dl34kwrcs69DrwO0Li8uYyPmjs55/jss9/o3XsGf/xxAIDrr69BVFRej5OJiEhGCGaRXwDEmFklfMX9ZqBL4Axm1hD4L9DeObcriFnkNCtX7uKRR6Yxa9YfANSpU5IxY9rRunVlj5OJiEhGCVqRd84lmFk3YDoQCkx0zq0ys2eBhc65qcALQAHgI//TyTY7564LVib5y9Chc5g16w+KFIlg8OAr6Nq1MWFhum2CiEhOYs5lr9bvxuXNLdySvTJnBQkJSezceYRy5QoCsHnzQUaOnMfTT19OsWL5PE4nIiJnY2aLnHON07WsinzO9913f9Cz5zTy5g3l11/vIyREz3QXEckuzqfIq302B/vjj/3ccMMHtG79FitX7mLfvuNs3nzQ61giIpJJssQldJKxjhw5yfPPz2HUqJ85cSKR/PnD6d//Uh59tBkREfovFxHJLfQXP4dJSnI0a/YGK1f6Lla47bZ6PP986+Rz8SIiknuoyOcwISHGffddyDvvLGfs2PY0a1Y+9YVERCRHUse7bG7HjiP07z+L2rVL0Lv3JQAkJiZhZupgJyKSA5xPxzsdyWdTJ04kMG7cLwwe/COHD5+kePF8PPTQRURGhhMaqv6UIiKiIp/tOOf46qv19Oo1nQ0b9gFw7bXVGDXqSiIjwz1OJyIiWYmKfDZy4EAcN9/8MdOn/w5AjRrFGTOmnR4iIyIiZ6Qin40ULJiXAwfiKFQoL4MGteThhy8iPDzU61giIpJFqchnYYmJSbzxxhLatKlM5cpFCAkx3nrreooUiaBEifxexxMRkSxOPbSyqB9//JPGjSfQteuXPPbYjOTx1aoVU4EXEZE00ZF8FrN580H69PmWDz9cBcAFFxTi5pvreJxKRESyIxX5LOLYsXheeGEuw4fP5fjxBCIjw+jXrwWPPXYJ+fKp17yIiJw7FfksYvPmgwwZMoeEhCQ6d67NiBFtueCCQl7HEhGRbExF3kNr1+6hWrVimBk1ahRn1KgradiwNJdeWsHraCIikgOo450Hdu8+yv33f0HNmi/z6adrksf36NFUBV5ERDKMjuQzUXx8IuPH/8ozz/zAwYMnCAsLSb5rnYiISEZTkc8k06ZtoFev6fz22x4A2revyujR7ahRo7jHyUREJKdSkc8E7767nFtvnQJATExRRo9ux9VXx2Cmp8SJiEjwqMgHiXMuuYhff31N6tady+2316dHj6bkyaNb0YqISPCpyGewpCTHpElLGTNmPnPm3EWhQhHkyxfO0qUP6PnuIiKSqdS7PgPNm7eFJk0mcM89U1mxYheTJi1NnqYCLyIimU1H8hkgNvYQjz8+k8mTVwBQrlwUI0a05ZZbdDtaERHxjor8eZo4cQndu3/DsWPx5M0bSp8+l9CvXwvy58/jdTQROQ/x8fHExsYSFxfndRTJJSIiIoiOjiY8PONuZa4if54uuKAQx47Fc+ONNXnhhbZUqlTE60gikgFiY2OJioqiYsWKuhJGgs45x969e4mNjaVSpUoZtl6dkz9Hy5fvZPTon5OH27SpzLJlD/DxxzepwIvkIHFxcRQrVkwFXjKFmVGsWLEMbznSkXwa7d17jIEDZ/Paa4tISnK0aHEBF11UDoB69Up5nE5EgkEFXjJTMD5vKvKpSEhI4rXXFjJw4Gz2748jNNTo3r0JVaoU9TqaiIhIitRcn4JZszbSoMFrdO/+Dfv3x9G6dSWWLn2AceOuomjRSK/jiUgOFxoaSoMGDahTpw7XXnstBw4cSJ62atUqWrVqRfXq1YmJiWHw4ME455Knf/PNNzRu3JhatWrRsGFDevfu7cUupGjJkiXcc889Xsc4qx9//JELL7yQsLAwPv7447POt2jRIurWrUvVqlXp0aNH8v/Dvn37aNu2LTExMbRt25b9+/cD8OWXXzJw4MBM2QcV+RS8884KVq3aTaVKhZkypTPffnsbdeqU9DqWiOQSkZGRLF26lJUrV1K0aFFefvllAI4fP851111Hv379WLt2LcuWLWPevHm88sorAKxcuZJu3brxzjvvsHr1ahYuXEjVqlUzNFtCQsJ5r2Po0KH06NEjU7d5Li644AImTZpEly5dUpzvwQcfZMKECaxfv57169czbdo0AIYNG0br1q1Zv349rVu3ZtiwYQB06NCBL774gmPHjgV9H9RcH+DIkZNs3XqI6tV9D40ZOrQV1asX45FHLiYiQm+VSK41Kkjn5nu71Ofxa9asGcuXLwdg8uTJNG/enCuvvBKAfPnyMX78eFq2bMnDDz/MiBEjePLJJ6lRowbgaxF48MEH/7HOI0eO0L17dxYuXIiZ8fTTT3PjjTdSoEABjhw5AsDHH3/Ml19+yaRJk7jzzjuJiIhgyZIlNG/enE8//ZSlS5dSuHBhAGJiYvjpp58ICQnhgQceYPPmzQCMGTOG5s2b/23bhw8fZvny5dSvXx+AX3/9lZ49exIXF0dkZCRvvvkm1atXZ9KkSXz66accOXKExMREvv76a7p3787KlSuJj49n0KBBdOzYkU2bNnHbbbdx9OhRAMaPH88ll1yS5vf3TCpWrAhASMjZj4e3b9/OoUOHuPjiiwG4/fbb+eyzz7jqqqv4/PPP+f777wG44447aNmyJcOHD8fMaNmyJV9++SU33XTTeWVMjSoXvlvRvvvuch5/fCZFi0aydOkDhIWFUKZMFP36tfA6nojkcomJicyaNSu5aXvVqlU0atTob/NUqVKFI0eOcOjQIVauXJmm5vnBgwdTqFAhVqzw3cjrVHNySmJjY5k3bx6hoaEkJiYyZcoU7rrrLn755RcqVKhAqVKl6NKlC7169aJFixZs3ryZdu3asWbNmr+tZ+HChdSp89cNw2rUqMGcOXMICwtj5syZ9O/fn08++QSAxYsXs3z5cooWLUr//v1p1aoVEydO5MCBAzRp0oQ2bdpQsmRJvv32WyIiIli/fj233HILCxcu/Ef+Sy+9lMOHD/9j/MiRI2nTpk2q+3+6rVu3Eh0dnTwcHR3N1q1bAdi5cydlypQBoHTp0uzcuTN5vsaNGzNnzhwV+WD79det9Ow5jfnzYwEoX74Qu3YdpWzZKI+TiUiWcQ5H3Bnp+PHjNGjQgK1bt1KzZk3atm2boeufOXMm77//fvJwkSKpXwbcqVMnQkN9D9nq3Lkzzz77LHfddRfvv/8+nTt3Tl7v6tWrk5c5dOgQR44coUCBAsnjtm/fTokSJZKHDx48yB133MH69esxM+Lj45OntW3blqJFfZ2dZ8yYwdSpUxk5ciTgu9Rx8+bNlC1blm7durF06VJCQ0NZt27dGfPPmTMn1X0MBjP7W+/5kiVLsm3btqBvN9cW+e3bD9O//3fJ95cvXboAw4e34dZb6+k+8yKSJZw6J3/s2DHatWvHyy+/TI8ePahVqxY//vjj3+bduHEjBQoUoGDBgtSuXZtFixYlN4Wfq8BidPp12/nz509+3axZMzZs2MDu3bv57LPPeOqppwBISkpi/vz5REREpLhvgeseMGAAV1xxBVOmTGHTpk20bNnyjNt0zvHJJ59QvXr1v61v0KBBlCpVimXLlpGUlHTWbWf0kXy5cuWIjY1NHo6NjaVcOd/l1aVKlWL79u2UKVOG7du3U7LkX326Tp2WCLZc2fEuMTGJSy99k0mTlpInTyiPP96cdeu6cfvt9VXgRSTLyZcvH+PGjWPUqFEkJCTwn//8h59++omZM2cCviP+Hj160LdvXwD69OnD0KFDk49mk5KSeO211/6x3rZt2yZ35oO/mutLlSrFmjVrSEpKYsqUKWfNZWZcf/31PProo9SsWZNixYoBcOWVV/LSSy8lz7d06dJ/LFuzZk02bNiQPHzw4MHk4jhp0qSzbrNdu3a89NJLyT3YlyxZkrx8mTJlCAkJ4e233yYxMfGMy8+ZM4elS5f+4yc9BR6gTJkyFCxYkPnz5+Oc46233qJjx44AXHfddfzvf/8D4H//+1/yeIB169b97XRFsOSaIu+cIzExCYDQ0BAef7w5111XnVWrHmLYsDZEReX1OKGIyNk1bNiQevXq8d577xEZGcnnn3/OkCFDqF69OnXr1uWiiy6iW7duANSrV48xY8Zwyy23ULNmTerUqcPGjRv/sc6nnnqK/fv3U6dOHerXr8/s2bMBX6/wa665hksuuST5nPLZdO7cmXfeeSe5qR5g3LhxLFy4kHr16lGrVq0zfsGoUaMGBw8eTD6q7tu3L0888QQNGzZMsRf9gAEDiI+Pp169etSuXZsBAwYA8NBDD/G///2P+vXr89tvv/3t6D+9FixYQHR0NB999BFdu3aldu3aydMaNGiQ/PqVV17h3nvvpWrVqlSpUoWrrroKgH79+vHtt98SExPDzJkz6devX/Iys2fPpkOHDuedMTUWeF1ldtC4vLmFW84t8+rVu+nVazotWpRnwIDLAV/R192sRORs1qxZQ82aNb2OkaONHj2aqKgo7r33Xq+jZKqdO3fSpUsXZs2a9Y9pZ/rcmdki51zj9GwrRx/J799/nJ49v6FevVeZMeN3/vvfRZw44fuGqAIvIuKtBx98kLx5c18r6ubNmxk1alSmbCtHdrxLTExiwoTFPPXUd+zde5yQEOOBBxrx7LNXkDdvjtxlEZFsJyIigttuu83rGJnuoosuyrRt5biKt2fPMVq3fovly33XI15+eQXGjm1P/fqlPU4mItmNTutJZgrG6fMcV+SLFYukSJEIKlQoxMiRV3LjjTX1Syoi5ywiIoK9e/fqcbOSKU49Tz6lyw7TI9t3vDt69CQjRszlllvqUqOG73a0W7ceomjRSCIjw72KKSLZXHx8PLGxsRn+fG+Rs4mIiCA6Oprw8L/XrvPpeBfUI3kzaw+MBUKB/3PODTttel7gLaARsBfo7JzblJZ1O+f44INV9OnzLbGxh1iwYBtff/0fAMqVK5iBeyEiuVF4eDiVKlXyOobIeQlakTezUOBloC0QCywws6nOudUBs90D7HfOVTWzm4HhQOd/ru3vFi/eTo8e3zB37hYALrywDP37X5rh+yAiIpKdBfNIvgmwwTm3EcDM3gc6AoFFviMwyP/6Y2C8mZlL4RzCn/sL0bjx6zgHJUvmZ+jQVtx5ZwNCQ3P01YAiIiLnLJhFvhywJWA4Fmh6tnmccwlmdhAoBuw520r3Hs1HaFgIPXs2ZcCAyyhUKGM7KYiIiOQU2aJ3vZndD9zvHzyRkDBw5ahRkEn3EsiNipPCFy3JMHqfg0/vcfDpPQ6+6qnPcmbBLPJbgfIBw9H+cWeaJ9bMwoBC+Drg/Y1z7nXgdQAzW5jeXoaSNnqPM4fe5+DTexx8eo+Dz8wWpnfZYJ7IXgDEmFklM8sD3AxMPW2eqcAd/tf/Br5L6Xy8iIiIpF3QjuT959i7AdPxXUI30bn/b+/+Y62u6ziOP18ZioDDNqqZS1hLsjt1hFS2huhg5GBhDJUo5yjWGiWtMJcrZ80f9IN0063NkNilMkVZOQoNlUGXKT9sXLj8KBmFc84Kt4xCMEnf/fH5nDq7nnvO98K995zzva/Hdna/55zP9/t9n/c9O+/z/Xy/5/OJfZJuA34fEeuAnwA/k3QQ+Dvpi4CZmZkNgEE9Jx8RjwGP9Xrs1qrl14Br+rnZFQMQmtXnHA8N53nwOceDzzkefCed47Yb8c7MzMyK8Y/LzczMSqpli7ykKyU9J+mgpJtrPH+GpDX5+e2SJgx9lO2tQI6XStovqUfSRknjmxFnO2uU46p28ySFJF+lfBKK5FnStfn9vE/SL4Y6xnZX4PPiPEmbJHXnz4xZzYiznUlaJemwpL19PC9J9+b/QY+kyQ03GhEtdyNdqPcn4H3A6cBuoKNXmy8B9+XlTwNrmh13O90K5vgKYFReXuwcD3yOc7uzgC5gGzCl2XG3263ge/l8oBt4R77/rmbH3U63gjleASzOyx3A+4N9IAAABjpJREFU882Ou91uwGXAZGBvH8/PAh4HBFwKbG+0zVY9kv/fkLgR8TpQGRK32lXA6ry8FpguzwfZHw1zHBGbIuJYvruNNNaBFVfkfQxwO2neBk93dnKK5PkLwI8i4hWAiDg8xDG2uyI5DqAyO9hY4KUhjK8UIqKL9EuzvlwF/DSSbcDZks6pt81WLfK1hsQ9t682EfEfoDIkrhVTJMfVFpG+QVpxDXOcu9veGxHrhzKwkinyXp4ITJT0tKRteYZMK65Ijr8DXCfpRdKvqpYMTWjDSn8/t9tjWFtrLknXAVOAac2OpUwkvQ24G1jY5FCGg7eTuuwvJ/VIdUm6KCL+0dSoymUB0BkRd0n6GGkMlAsj4s1mBzacteqRfH+GxKXekLjWpyI5RtIM4FvAnIj49xDFVhaNcnwWcCGwWdLzpHNs63zxXb8VeS+/CKyLiBMRcQg4QCr6VkyRHC8CHgaIiK3ASNK49jZwCn1uV2vVIu8hcQdfwxxL+hDwY1KB9znM/qub44g4EhHjImJCREwgXfcwJyJOepzqYarI58WjpKN4JI0jdd//eSiDbHNFcvwCMB1A0gdJRf7lIY2y/NYB1+er7C8FjkTEX+qt0JLd9eEhcQddwRwvB8YAj+RrGl+IiDlNC7rNFMyxnaKCed4AzJS0H3gDuCki3PNXUMEc3wjcL+lrpIvwFvrAq38kPUj6MjouX9vwbWAEQETcR7rWYRZwEDgGfK7hNv0/MDMzK6dW7a43MzOzU+Qib2ZmVlIu8mZmZiXlIm9mZlZSLvJmZmYl5SJv1gSS3pC0q+o2oU7bowOwv05Jh/K+duYRyfq7jZWSOvLyN3s998ypxpi3U8nLXkm/lnR2g/aTPNuZWd/8EzqzJpB0NCLGDHTbOtvoBH4TEWslzQR+GBEXn8L2TjmmRtuVtBo4EBF31mm/kDRz3w0DHYtZGfhI3qwFSBojaWM+yt4j6S2z1Uk6R1JX1ZHu1Pz4TElb87qPSGpUfLuA9+d1l+Zt7ZX01fzYaEnrJe3Oj8/Pj2+WNEXS94AzcxwP5OeO5r8PSZpdFXOnpKslnSZpuaRn8zzYXyyQlq3kyTckfSS/xm5Jz0j6QB557TZgfo5lfo59laQduW2tWf/Mho2WHPHObBg4U9KuvHwIuAaYGxH/zMOubpO0rteIYZ8BNkTEnZJOA0bltrcAMyLiVUnfAJaSil9fPgnskXQJacSsj5Lmp94u6XekOcNfiojZAJLGVq8cETdLuiEiJtXY9hrgWmB9LsLTgcWkcc2PRMSHJZ0BPC3piTyO/Fvk1zedNLIlwB+BqXnktRnAsoiYJ+lWqo7kJS0jDXH9+dzVv0PSUxHxap18mJWWi7xZcxyvLpKSRgDLJF0GvEk6gn038NeqdZ4FVuW2j0bELknTgA5S0QQ4nXQEXMtySbeQxhNfRCqiv6oUQEm/BKYCvwXukvR9Uhf/ln68rseBe3IhvxLoiojj+RTBxZKuzu3GkiaI6V3kK19+zgX+ADxZ1X61pPNJQ6aO6GP/M4E5kr6e748EzsvbMht2XOTNWsNngXcCl0TECaVZ6UZWN4iIrvwlYDbQKelu4BXgyYhYUGAfN0XE2sodSdNrNYqIA0rz3M8C7pC0MSLq9QxUr/uapM3AJ4D5wEOV3QFLImJDg00cj4hJkkaRxkn/MnAvcDuwKSLm5osUN/exvoB5EfFckXjNys7n5M1aw1jgcC7wVwDjezeQNB74W0TcD6wEJpNmrvu4pMo59tGSJhbc5xbgU5JGSRoNzAW2SHoPcCwifk6apGhyjXVP5B6FWtaQTgNUegUgFezFlXUkTcz7rCkijgFfAW7U/6eSrkypubCq6b9IU/ZWbACWKHdrKM2kaDZsucibtYYHgCmS9gDXk85B93Y5sFtSN+ko+Z6IeJlU9B6U1EPqqr+gyA4jYifQCewAtgMrI6IbuIh0LnsXaRasO2qsvgLoqVx418sTwDTgqYh4PT+2EtgP7JS0lzSFcd2exBxLD7AA+AHw3fzaq9fbBHRULrwjHfGPyLHty/fNhi3/hM7MzKykfCRvZmZWUi7yZmZmJeUib2ZmVlIu8mZmZiXlIm9mZlZSLvJmZmYl5SJvZmZWUi7yZmZmJfVfUJ8334Kd13wAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 576x360 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}