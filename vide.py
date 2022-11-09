import matplotlib.pyplot as plt
import numpy as np


class Vide:

    def _has_same_length(a, b):
        """
        Check if a and b has the same dimesions

        Parameters
        ----------
            a : array_like

            b : array_like

        Returns
        -------
            True if len(a) == len(b) else False
        """
        if (len(a) != len(b)):
            return False
        return True

    def HPDI(posteriori_samples, credible_mass=0.89):
        """
        Calculate the highest posterior density interval (HPDI).

        Choosing the narrowest interval, which for a unimodal
        distribution will involve choosing those values of highest
        probability density including the mode (the maximum a posteriori).
        This is sometimes called the highest posterior density interval (HPDI).

        Parameters
        ----------
        posteriori_samples : array_like
            Samples of distribution of probability.

        credible_mass : float [0, 1], optional
            Value float between (0, 1) that define highest posterior
            density interval (HPDI)

        Returns
        -------
        hpdi : A tuple with two positions (HPDI_min, HPDI_max)
        """

        if credible_mass > 1 or credible_mass <= 0:
            print('The credible mass must be between 0 and 1')
            return False

        if credible_mass < 0.70:
            print('Warning: Very low values of the credibility interval can \
                  make this operation take a long time to complete.')

        sorted_points = sorted(posteriori_samples)
        ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
        nCIs = len(sorted_points) - ciIdxInc
        ciWidth = [0]*nCIs

        for i in range(0, nCIs):
            ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
            HPDI_min = sorted_points[ciWidth.index(min(ciWidth))]
            HPDI_max = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]

        return(HPDI_min, HPDI_max)

    def plot_lines(alpha, beta, range_limits=(-2, 2), title=None,
                   xlabel=None, ylabel=None, linewidth=0.2, figsize=(17, 9),
                   ylim=(-2, 2), xlim=(-2, 2)):
        """
        Plot the probabilistic lines.

        Parameters
        ----------
        alpha(𝛼) : array_like
            Samples of the intercept (𝛼) of the probability distribution.

        beta(𝛽) : array_like
            Samples of slopes (𝛽) of the probability distribution.

        range_limits : tuple(min_value=-2, max_value=2), optional
            A tuple with two values, minimum and maximum of the range.

        title : string, optional
            Title of graph

        xlabel : string, optional
            Axis X label

        ylabel : string, optional
            Axis Y label

        linewidth : float, optional
            The width of the line

        figsize : tuple, optional
            Set the figsize of the figure

        xlim : tuple
            Set the plot xlim

        ylim : tuple
            Set the plot ylim

        Returns
        -------
        Plot of the graph line.
        """
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = 'lightgray'

        if not Vide._has_same_length(alpha, beta):
            print('Error: Length of alpha and beta are no equal')
            return False

        range_full = np.linspace(range_limits[0], range_limits[1])

        mu = [alpha + beta * x for x in range_full]

        plt.figure(figsize=figsize)

        plt.plot(range_full, mu, color='darkblue', linewidth=linewidth)

        if title:
            plt.title(title)

        if xlabel:
            plt.xlabel(xlabel)

        if ylabel:
            plt.ylabel(ylabel)

        plt.grid(ls='--', color='white', alpha=0.4)

        plt.ylim(ylim)
        plt.xlim(xlim)

    def mu_lm(alpha, beta, range_limits=(-2, 2)):
        """
        Calculate the line average (μ) of the bivariate regression.

        $$ μ = mean(𝛼) + mean( 𝛽 ) * range(x) $$

        Parameters
        ----------
        alpha(𝛼) : array_like
            Samples of the intercept (𝛼) of the probability distribution.

        beta(𝛽) : array_like
            Samples of slopes (𝛽) of the probability distribution.

        range_limits : tuple(min_value=-2, max_value=2), optional
            A tuple with two values, minimum and maximum of the range.


        Returns
        -------
        Posteriori line average of the linear model.

        """
        if not Vide._has_same_length(alpha, beta):
            print('Error: Length of alpha and beta are no equal')
            return False

        range_full = np.linspace(range_limits[0], range_limits[1])

        average_line = np.mean(alpha) + np.mean(beta) * range_full

        return average_line

    def CI_lm(alpha, beta, range_limits=(-2, 2), credible_mass=0.89):
        """
        Calculate the bayesian compatibility interval of bivariate regression.

        $$ μ = mean(𝛼) + mean( 𝛽 ) * range(x) $$

        Parameters
        ----------
        alpha(𝛼) : array_like
            Samples of the intercept (𝛼) of the probability distribution.

        beta(𝛽) : array_like
            Samples of slopes (𝛽) of the probability distribution.

        range_limits : tuple(min_value=-2, max_value=2), optional
            A tuple with two values, minimum and maximum of the range.

        credible_mass : float [0, 1], optional
            Value float between (0, 1) that define highest posterior
            density interval (HPDI)

        Returns
        -------
        Compatibility Interval of the linear model.

        """
        if not Vide._has_same_length(alpha, beta):
            print('Error: Length of alpha and beta are no equal')
            return False

        range_full = np.linspace(range_limits[0], range_limits[1])

        posterioris = np.array([[alpha + beta * x for x in range_full]])[0]

        CI = np.array([Vide.HPDI(post, credible_mass) for post in posterioris])

        return CI

    def plot_lm(outcome, predictor, alpha, beta, title=None,
                xlabel=None, ylabel=None, linewidth=0.2, figsize=(17, 9),
                ylim=(-2, 2), xlim=(-2, 2)):
        """
        Plot posterioris of the linear model.

        Parameters
        ----------
        outcome : array_like
            The variable outcome of linear model

        predictor : array_like
            The variable preditor of linear model

        alpha : array_like
            Posteriori of alpha (𝛼)

        beta : array_like
            Posteriori of beta (𝛽)

        range_limits : tuple(min_value=-2, max_value=2), optional
            A tuple with two values, minimum and maximum of the range.

        title : string, optional
            Title of graph

        xlabel : string, optional
            Axis X label

        ylabel : string, optional
            Axis Y label

        linewidth : float, optional
            The width of the line

        figsize : tuple, optional
            Set the figsize of the figure

        xlim : tuple
            Set the plot xlim

        ylim : tuple
            Set the plot ylim


        Returns
        -------
        Plot tie graph with the mean and CI and the points original data raw.
        """
        if not Vide._has_same_length(outcome, predictor):
            print('Error: Length of outcome and predictor are no equal')
            return False

        if not Vide._has_same_length(alpha, beta):
            print('Error: Length of alpha and beta are no equal')
            return False

        # Parameters
        min_range = min(predictor)
        max_range = max(predictor)
        mu_range = np.linspace(min_range, max_range)
        mu_line = Vide.mu_lm(alpha, beta, range_limits=(min_range, max_range))
        CI = Vide.CI_lm(alpha, beta, range_limits=(min_range, max_range))

        # Plot data
        plt.scatter(predictor, outcome)

        # Plot mu
        plt.plot(mu_range, mu_line, color='black')

        # Plot CI
        plt.fill_between(mu_range, CI[:, 0], CI[:, 1], color='gray', alpha=0.4)

        plt.grid(ls='--', color='white', linewidth=0.4)

        if title:
            plt.title(title)

        if xlabel:
            plt.xlabel(xlabel)

        if ylabel:
            plt.ylabel(ylabel)

        plt.grid(ls='--', color='white', alpha=0.4)
