import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';

import { Icon, Flex, SimpleGrid, HStack, VStack, Button, ChakraProvider,
  Box, FormControl, FormLabel, Select, Center, Alert, AlertIcon, Progress, useColorMode, IconButton } from '@chakra-ui/react';
import { FaMoon, FaSun } from 'react-icons/fa'
import { generateTheme } from 'catppuccin-chakra-ui-theme'

// @ts-ignore
import styles from './App.module.scss';

const WIDTH = 600;
const HEIGHT = 450;
const SERVER = "ws://localhost:8080";

function App() {
  const [scene, setScene] = useState('cornell_box')
  const [rendering, setRendering] = useState(false);
  const [pixelsRendered, setPixelsRendered] = useState(0);
  const { toggleColorMode, colorMode } = useColorMode();
  const canvasRef = useRef(null);
  const options = [
    ["Cornell Box", "cornell_box"],
    ["Cubes", "cubes"],
    ["Flying Unicorn", "flying_unicorn"],
  ];

  let image: Uint8ClampedArray;
  let frameDidChange: boolean;

  const { connection, error, send } = useWebSocket({
    url: SERVER,
    binaryType: 'arraybuffer',
    onMessage: event => {
      const view = new DataView(event.data);
      switch (view.getUint8(0)) {
        case ServerMessage.RenderedPixels:
          const x = view.getUint16(1, true);
          const y = view.getUint16(3, true);
          const i = 4 * (y * WIDTH + x);
          image[i] = view.getUint8(5);
          image[i + 1] = view.getUint8(6);
          image[i + 2] = view.getUint8(7);
          image[i + 3] = 255;
          frameDidChange = true;
          setPixelsRendered(pixelsRendered + 1);
          break;

        case ServerMessage.DoneRendering:
          setRendering(false);
          break;
      }
    }
  });

  function drawPixel(ctx: CanvasRenderingContext2D, x: number, y: number, [r, g, b]: Color) {
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.fillRect(x, y, 1, 1);
  }

  let ctx: CanvasRenderingContext2D | undefined;

  useEffect(() => {
    if (error)
      console.error('Timgjdsafisd', error);
  }, [error])

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      ctx = canvas.getContext("2d");
      ctx!.strokeStyle = 'none';
      image = new Uint8ClampedArray( Array(4 * WIDTH * HEIGHT).map(() => 255) );
      frameDidChange = true;
      (function animate() {
        if (frameDidChange) {
          const frame = new ImageData(image, WIDTH, HEIGHT);
          ctx?.putImageData(frame, 0, 0);
          frameDidChange = false;
        }
        requestAnimationFrame(animate);
      })()
    }
  }, [canvasRef]);

  return (
    <SimpleGrid columns={2}>
      <ColorModeToggle onClick={toggleColorMode} />
      <Center h='100vh'>
        <VStack spacing={5}>
          <canvas
            ref={canvasRef}
            width={WIDTH}
            height={HEIGHT}
            style={{
              boxShadow:
                colorMode === 'light' ?
                  '0px 8px 15px lightgrey' :
                  '0px 8px 100px #FFF3',
            }}
            className={styles.renderResult}
          />
          <Progress size='lg' hasStripe value={80}/>
        </VStack>
      </Center>
      <Box
        as='form'
        m='20px'
        onSubmit={e => {
          e.preventDefault();
          if (connection === 'open') {
            if (rendering) {
              setRendering(false);
              send(JSON.stringify({
                type: 'stop_rendering'
              }))
            } else {
              setRendering(true);
              send(JSON.stringify({
                type: 'render',
                scene,
              }));
            }
          }
        }}
      >
        <VStack>
          <FormControl>
            <FormLabel>Scene</FormLabel>
            <Select
              value={scene}
              onChange={e => setScene(e.target.value)}
            >
              {options.map(([name, value], i) => (
                <option key={`opt${i}`} value={value}>{name}</option>
              ))}
            </Select>
          </FormControl>
          <Button type="submit">{rendering ? 'Stop' : 'Render'}</Button>
          {rendering ?
            <p>Rendering ({Math.floor(pixelsRendered / (WIDTH * HEIGHT) * 100)}%)</p> :
            null
          }
          {error !== null ?
            <Alert status='error' variant='left-accent'>
              <AlertIcon/>
              Could not connect to the server.
            </Alert> :
            null
          }
        </VStack>
      </Box>
    </SimpleGrid>
  )
}

enum ServerMessage {
  RenderedPixels = 0,
  DoneRendering = 1
}

async function sleep(ms: number) {
  await new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

export type ConnectionState = 'connecting' | 'open' | 'closed'

export function useWebSocket({
  url,
  protocols,
  binaryType,
  onMessage
}: {
  url: URL | string,
  protocols?: string | string[],
  binaryType?: 'blob' | 'arraybuffer',
  onMessage?: (m: MessageEvent) => void,
}) {
  const [socket, setSocket] = useState(null);
  const [connection, setConnection] = useState('connecting' as ConnectionState);
  const [error, setError] = useState(null);

  useEffect(() => {
    let socket: WebSocket;
    try {
      socket = new WebSocket(url, protocols);
    } catch (err) {
      setError(err);
      setConnection('closed');
      return;
    }

    if (binaryType)
      socket.binaryType = binaryType;

    socket.addEventListener('open', event => {
      setConnection('open');
    });

    socket.addEventListener('message', event => {
      onMessage?.(event);
    });

    socket.addEventListener('error', event => {
      setError(event);
    });

    socket.addEventListener('close', event => {
      setConnection('closed');
      if (!event.wasClean)
        setError(event);
    });

    setSocket(socket);
  }, []);

  const send = (message: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    socket.send(message);
  }

  return {
    connection,
    error,
    send,
  }
}

function assert(cond: boolean): asserts cond {
  if (cond)
    throw new Error('assertion failed');
}

type Color = [number, number, number];

function ColorModeToggle({ isDark, onClick }) {
  return (
    <IconButton
      variant='solid'
      aria-label='Color mode toggle'
      rounded='full'
      size='sm'
      position='absolute'
      top={4}
      left={4}
      icon={<Icon as={isDark ? FaMoon : FaSun}/>}
      onClick={onClick}
    />
  )
}

const theme = generateTheme('latte', 'mocha');

createRoot(document.getElementById('root'))
  .render(
    <ChakraProvider theme={theme}>
      <App/>
    </ChakraProvider>
  );