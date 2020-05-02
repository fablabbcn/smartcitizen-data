def scatter_diagram(fig, gs, n, dataframeTrain, dataframeTest):
    ## TODO DOCUMENT

    ax = fig.add_subplot(gs[n])

    plt.plot(dataframeTrain['reference'], dataframeTrain['prediction'], 'go', label = 'Train ' + model_name, alpha = 0.3)
    plt.plot(dataframeTest['reference'], dataframeTest['prediction'], 'bo', label = 'Test ' + model_name, alpha = 0.3)
    plt.plot(dataframeTrain['reference'], dataframeTrain['reference'], 'k', label = '1:1 Line', linewidth = 0.4, alpha = 0.3)

    plt.legend(loc = 'best')
    plt.ylabel('Prediction (-)')
    plt.xlabel('Reference (-)')