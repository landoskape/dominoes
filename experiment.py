from matplotlib.pyplot import show
from dominoes.experiments import create_experiment

if __name__ == "__main__":

    # Create experiment
    exp = create_experiment()

    if exp.args.showprms:
        # Load saved experiment (just experiment parameters)
        _ = exp.load_experiment(no_results=True)

        # Report parameters of saved experiment
        exp.report(args=True)

    elif not exp.args.justplot:
        # Report experiment details
        exp.report(init=True, args=True, meta_args=True)

        # Run main experiment method
        results, nets = exp.main()

        # Save results if requested
        if not exp.args.nosave:
            exp.save_experiment(results)

            # Save copy of repo
            exp.save_repo()

            # Save networks
            if exp.args.save_networks:
                exp.save_networks(nets)

    else:
        # Load saved experiment (parameters and results)
        print("Loading saved experiment...")
        results = exp.load_experiment()

        # Report saved experiment parameters
        exp.report(args=True)

    # Plot results unless calling script to show saved parameters
    if not exp.args.showprms:
        exp.plot(results)

        # Show all figures at end if requested
        if exp.args.showall:
            show()
