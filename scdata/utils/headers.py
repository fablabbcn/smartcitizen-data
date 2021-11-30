def process_headers(headers):
    result = {}
    if 'total' in headers: result['total_pages'] = headers['total']
    if 'per-page' in headers: result['per_page'] = headers['per-page']
    if 'link' in headers:
        for item in headers.get('link').split(','):
            chunk = item.replace(' ', '').split(';')
            if 'rel' in chunk[1]:
                which = chunk[1].replace('"', '').split('=')[1]
                if which == 'next':
                    result['next'] = chunk[0].strip('<').strip('>')
                elif which == 'last':
                    result['last'] = chunk[0].strip('<').strip('>')
                elif which == 'prev':
                    result['prev'] = chunk[0].strip('<').strip('>')
                elif which == 'first':
                    result['first'] = chunk[0].strip('<').strip('>')
    return result
