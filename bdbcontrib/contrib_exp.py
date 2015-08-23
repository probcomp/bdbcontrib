from bayeslite.shell.hook import bayesdb_shell_cmd
from bdbcontrib.general_utils import ArgparseError, ArgumentParser

import bdbexp

experiments = {
    'haystacks' : bdbexp.haystacks,
    'hyperparams' : bdbexp.hyperparams,
    'infer' : bdbexp.infer,
    'kl_divergence' : bdbexp.kl_divergence,
    'permute' : bdbexp.permute,
    'predictive_pdf' : bdbexp.predictive_pdf,
    'recover' : bdbexp.recover
    }

@bayesdb_shell_cmd('experiment')
def experiment(self, argin):
    '''
    Launch an experimental inference quality test.
    USAGE: .experiment <exp_name> [exp_args ...]

    <exp_name>
        permute         kl_divergence
        haystacks       predictive_pdf
        hyperparams     recover
        infer

    [exp_args]
        To see experiment specific arguments, use
        .experiment <exp_name> --help

    Examples:
    bayeslite> .experiment predictive_pdf --help
    bayeslite> .experiment haystacks --n_iter=200 --n_distractors=4
    '''
    expname = argin.split()[0] if argin != '' else argin
    if expname not in experiments:
        print 'Invalid experiment {}'.format(expname)
        print 'For help use: .help experiment'
        return

    try:
        experiments[argin.split()[0]].main(argin.split()[1:], halt=True)
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return
    except SystemExit:
        return  # This happens when --help is invoked.
