import { afterEach, describe, expect, it, vi } from 'vitest'

import { checkHermesUpdate } from './system'

// The backend's update check runs bounded-but-serial git/GitHub probes and
// shares the process with maintenance work (profile deletes, gateway drains),
// so a busy-but-alive remote backend can answer after the 30s fetch default
// fires — surfacing "couldn't reach the update server" over a check that
// actually completed. The request must carry its own budget above that default.
describe('checkHermesUpdate request timeout', () => {
  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('sends an explicit timeoutMs above the default fetch budget', async () => {
    const api = vi.fn(() => Promise.resolve({ can_apply: false, behind: 0 }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

    await checkHermesUpdate(true)

    expect(api).toHaveBeenCalledOnce()
    const request = api.mock.calls[0][0] as { path?: string; timeoutMs?: number }
    expect(request.path).toBe('/api/hermes/update/check?force=true')
    expect(request.timeoutMs).toEqual(expect.any(Number))
    expect(request.timeoutMs).toBeGreaterThan(30_000)
  })
})
