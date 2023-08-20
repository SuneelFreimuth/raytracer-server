import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';

import {
  Grid, Icon, VStack, Button, ChakraProvider,
  Box, FormControl, FormLabel, Select, Center, Alert, AlertIcon, Progress, useColorMode, IconButton, GridItem, Table, Thead, Tr, Td, Tbody, HStack, Drawer, Heading, DrawerOverlay, DrawerContent, DrawerCloseButton, DrawerHeader, DrawerBody, Container, Text
} from '@chakra-ui/react';
import { FaImage, FaInfoCircle, FaMoon, FaStop, FaSun } from 'react-icons/fa';
// import { Canvas } from 'reaflow';

createRoot(document.getElementById('root'))
  .render(
    <ChakraProvider>
      <App />
    </ChakraProvider>
  );

const WIDTH = 600;
const HEIGHT = 450;
const SERVER = "ws://localhost:8080";

interface RenderResult {
  imageBitmap: ImageBitmap,
  /** In seconds with millisecond precision. */
  timeToRender: number,
}

interface RenderJob {
  start: number,
  pixelsRendered: number,
}

const options = [
  ["Cornell Box", "cornell_box"],
  ["Cubes", "cubes"],
  ["Flying Unicorn", "flying_unicorn"],
];

let image: Uint8ClampedArray;
let renderJob: RenderJob | null = null;

function App() {
  const [scene, setScene] = useState('cornell_box')
  const [renderResults, setRenderResults] = useState([] as RenderResult[]);
  const { toggleColorMode, colorMode } = useColorMode();
  const canvasRef = useRef(null as HTMLCanvasElement | null);

  const { connection, error, send } = useWebSocket({
    url: SERVER,
    binaryType: 'arraybuffer',
    onMessage: async (event) => {
      const view = new DataView(event.data);
      switch (view.getUint8(0)) {
        case ServerMessage.RenderedPixels:
          drawPixel(view);
          imageDidChange = true;

          const lastPixel = WIDTH * HEIGHT;
          if (renderJob!.pixelsRendered < lastPixel)
            renderJob!.pixelsRendered += 1;

          if (renderJob!.pixelsRendered === lastPixel) {
            const timeToRender = (Date.now() - renderJob!.start) / 1000;
            const imageBitmap = await createImageBitmap(new ImageData(image, WIDTH, HEIGHT));
            setRenderResults(rrs => {
              const renderResults = structuredClone(rrs);
              renderResults.unshift({ imageBitmap, timeToRender });
              return renderResults;
            });
            renderJob = null;
          }
          break;
      }
    }
  });

  let imageDidChange: boolean;

  const drawPixel = (view: DataView) => {
    const x = view.getUint16(1, true);
    const y = view.getUint16(3, true);
    const i = 4 * (y * WIDTH + x);
    image[i] = view.getUint8(5);
    image[i + 1] = view.getUint8(6);
    image[i + 2] = view.getUint8(7);
    image[i + 3] = 255;
  }

  let ctx: CanvasRenderingContext2D;

  useEffect(() => {
    if (canvasRef.current === null)
      return;
    ctx = canvasRef.current!.getContext("2d")!;
    ctx.strokeStyle = 'none';
    image = new Uint8ClampedArray(4 * WIDTH * HEIGHT).fill(255);
    imageDidChange = true;
    (function animate() {
      if (imageDidChange) {
        const frame = new ImageData(image, WIDTH, HEIGHT);
        ctx.putImageData(frame, 0, 0);
        imageDidChange = false;
      }
      requestAnimationFrame(animate);
    })();
  }, [canvasRef]);

  return (
    <Grid
      w='100vw'
      h='100vh'
      templateAreas={`
        "canvas controls"
        "canvas results"
      `}
      gridTemplateColumns='700px 1fr'
      gridTemplateRows='50vh 50vh'
    >
      <HStack position='absolute' top={4} left={4} spacing={4}>
        <About/>
        <ColorModeToggle isDark={colorMode === 'light'} onClick={toggleColorMode} />
      </HStack>
      <GridItem>
        <Center h='100vh'>
          <VStack spacing={25}>
            <canvas
              ref={canvasRef}
              width={WIDTH}
              height={HEIGHT}
              style={{
                boxShadow:
                  colorMode === 'light' ?
                    '0px 8px 15px #999' :
                    'none'
              }}
            />
            {Boolean(renderJob) ?
              <Progress
                value={renderJob!.pixelsRendered / (WIDTH * HEIGHT) * 100}
                style={{ width: '100%' }}
              /> :
              null
            }
          </VStack>
        </Center>
      </GridItem>
      <GridItem m='20px'>
        <Box
          as='form'
          onSubmit={e => {
            e.preventDefault();
            if (connection === 'open') {
              const request = JSON.stringify(
                Boolean(renderJob) ?
                  { type: 'stop_rendering' } :
                  { type: 'render', scene }
              );
              renderJob = {
                pixelsRendered: 0,
                start: Date.now(),
              };
              send(request);
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
            <Button type="submit" leftIcon={Boolean(renderJob) ? <FaStop /> : undefined}>
              {Boolean(renderJob) ? <>Stop</> : 'Render'}
            </Button>
            {error !== null ?
              <Alert status='error' variant='left-accent'>
                <AlertIcon/>
                Could not connect to the server.
              </Alert> :
              null
            }
          </VStack>
        </Box>
      </GridItem>
      <GridItem m='20px' colStart={2}>
        <Table>
          <Thead>
            <Tr>
              <Td w={120}><Icon as={FaImage}/></Td>
              <Td>Time to Render</Td>
            </Tr>
          </Thead>
          <Tbody>
            {renderResults.map(({ imageBitmap, timeToRender }, i) => (
              <Tr key={`row${i}`}>
                <Td>
                  <ImageBitmapView bitmap={imageBitmap}/>
                </Td>
                <Td>{timeToRender.toFixed(1)}s</Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </GridItem>
    </Grid>
  )
}

function About() {
  const [isOpen, setIsOpen] = useState(false);
  const btnRef = useRef(null as HTMLButtonElement | null);

  return (
    <>
      <IconButton 
        ref={btnRef}
        variant='solid'
        aria-label='Color mode toggle'
        rounded='full'
        size='sm'
        icon={<FaInfoCircle/>}
        onClick={() => {
          setIsOpen(true);
        }}
      />
      <Drawer
        isOpen={isOpen}
        placement='left'
        size='full'
        onClose={() => {
          setIsOpen(false);
        }}
        finalFocusRef={btnRef}
      >
        <DrawerOverlay/>
        <DrawerContent>
          <DrawerCloseButton
            position='relative'
            top={2}
            left={4}
          />
          <DrawerBody>
            <Container>
              <VStack spacing={4} align='left'>
                <Heading size='3xl'>About</Heading>
                <Text fontSize='xl'>Frontend for a raytracer written in Rust.</Text>
                <Heading size='lg'>How it Works</Heading>
                {/* <Canvas
                  maxWidth={800}
                  maxHeight={600}
                  nodes={[
                    {
                      id: '1',
                      text: '1'
                    },
                    {
                      id: '2',
                      text: '2'
                    }
                  ]}
                  edges={[
                    {
                      id: '1-2',
                      from: '1',
                      to: '2'
                    }
                  ]}
                /> */}
              </VStack>
            </Container>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </>
  )
}

function ColorModeToggle({ isDark, onClick }) {
  return (
    <IconButton
      variant='solid'
      aria-label='Color mode toggle'
      rounded='full'
      size='sm'
      icon={<Icon as={isDark ? FaSun : FaMoon} />}
      onClick={onClick}
    />
  )
}

function ImageBitmapView({ bitmap }: {
  bitmap: ImageBitmap,
}) {
  const canvasRef = useRef(null as HTMLCanvasElement | null);
  const width = 100;
  const height = width * bitmap.height / bitmap.width;

  useEffect(() => {
    (async () => {
      const resizedBitmap = await createImageBitmap(bitmap, {
        resizeWidth: width,
        resizeHeight: height
      });
      canvasRef
        .current
        ?.getContext('bitmaprenderer')
        ?.transferFromImageBitmap(resizedBitmap);
    })()
  }, [canvasRef, bitmap]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
    />
  );
}

enum ServerMessage {
  RenderedPixels = 0,
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
  const [socket, setSocket] = useState(null as WebSocket | null);
  const [connection, setConnection] = useState('connecting' as ConnectionState);
  const [error, setError] = useState(null as any);

  useEffect(() => {
    let sock: WebSocket;
    try {
      sock = new WebSocket(url, protocols);
    } catch (err) {
      setError(err);
      setConnection('closed');
      return;
    }

    if (binaryType)
      sock.binaryType = binaryType;

    sock.addEventListener('open', _ => {
      setConnection('open');
    });

    sock.addEventListener('message', event => {
      onMessage?.(event);
    });

    sock.addEventListener('error', event => {
      setError(event);
    });

    sock.addEventListener('close', event => {
      setConnection('closed');
      if (!event.wasClean)
        setError(event);
    });

    setSocket(sock);

    return () => {
      sock.close();
    };
  }, []);

  const send = (message: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    socket?.send(message);
  };

  return {
    connection,
    error,
    send,
  };
}

function assert(cond: boolean): asserts cond {
  if (cond)
    throw new Error('assertion failed');
}
