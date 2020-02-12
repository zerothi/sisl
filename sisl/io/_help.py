__all__ = ['starts_with_list']


def starts_with_list(l, comments):
    for comment in comments:
        if l.strip().startswith(comment):
            return True
    return False
