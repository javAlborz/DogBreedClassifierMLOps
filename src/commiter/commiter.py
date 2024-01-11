import subprocess
import time


def _run_git_command(*args):
    try:
        result = subprocess.run(['git', *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing `git {' '.join(args)}`:\n{e.stderr.decode().strip()}")
        exit(1)


def branch_and_commit():
    """Creates a new branch and commits the current state of the repository there.

    Returns:
        str: The message of the last commit.
    """
    time_id = int((time.time()*100)%100000000)
    stash_name = "stash_{:08d}".format(time_id)
    new_branch = "checkpoint/branch_{:08d}".format(time_id)
    merge_message = "merge_{:08d}".format(time_id)
    commit_message = "commit_{:08d}".format(time_id)

    # 1. Stash all changes into "stash_{rand_id}"
    stashed = False
    if _run_git_command('status', '--porcelain') != '':
        stashed = True
        _run_git_command('stash', 'push', '--include-untracked', '-m', stash_name)

    # 2. Create branch "branch_{rand_id}"
    _run_git_command('branch', new_branch)

    # 3. Merge the current branch into new_branch
    current_branch = _run_git_command('rev-parse', '--abbrev-ref', 'HEAD')
    _run_git_command('checkout', new_branch)
    _run_git_command('merge', '--strategy-option=theirs', current_branch, '-m', merge_message)

    # 4. Apply the stash in new_branch and commit all changes
    if stashed:
        _run_git_command('stash', 'apply', 'stash^{/'+stash_name+'}')
        if _run_git_command('status', '--porcelain') != '':
            _run_git_command('add', '.')
            _run_git_command('commit', '-m', commit_message)
    last_commit = _run_git_command('log', '-1', '--pretty="%s"')

    # 5. Go back to the initial branch and pop "auto_stash" there
    _run_git_command('checkout', current_branch)
    if stashed:
        _run_git_command('stash', 'pop')
    
    return last_commit

if __name__ == '__main__':
    print(branch_and_commit())